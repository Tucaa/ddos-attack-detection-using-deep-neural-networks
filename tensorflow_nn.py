import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, losses
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef
)
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from functions import *
from metrics_exporter import export_metrics_to_json

# Hiperparametri
SEQUENCE_LEN  = 50
BATCH_SIZE    = 64
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
DROPOUT       = 0.3
LEARNING_RATE = 1e-3
EPOCHS        = 20
N_SPLITS      = 5   # Broj foldova za TimeSeriesSplit

EXCLUDED_COLS = ['label', 'window_id', 'timestamp', 'ts_formated', 'attack_active', 'instance_id', 'vector_id']

LABELS = [
    'normal',
    'udp_flood_large',
    'dns_amplification', 
    'subnet_carpet_bombing', 
    'syn_flood',
    'icmp_flood', 
    'udp_flood_mixed',
    'ntp_amplification',
    'ack_flood'
]

# Moja verzija tf nije podrzavala treniranje na GPU 
# Moralo je preko wlsa da se podesi
# Pokusao sam preko dockera da resim ovaj problem ali kad pokrenem ovu skriptu iz docker okruzenja i dalje nije prepoznavao GPU 
# Nisam siguran zasto, jer ollama model bez problema koristi gpu.
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"Found gpu GPU: {gpus}")
    try:
        # 2. Ukljucivanje "Memory Growth" opcije
        # Ovo sprecava TF da odmah zauzme 100% VRAM-a, vec alocira memoriju po potrebi.
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth za GPU is enabled.")
    except RuntimeError as e:
        # Memory growth mora biti postavljen pre inicijalizacije GPU-a
        print(e)
else:
    print("GPU not found training on CPU.")

# Priprema podataka 
def prepare_data(csv_path: str, seq_len: int = SEQUENCE_LEN):
    """
    Ucitava CSV, sortira po timestamp-u, skalira feature i pravi sekvence.
    """
    try:
        df = pd.read_csv(csv_path)

        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        feature_cols = [c for c in df.columns if c not in EXCLUDED_COLS]
        X_raw = df[feature_cols].values.astype(np.float32)

        label_enc = LabelEncoder()
        label_enc.classes_ = np.array(LABELS)
        Y_raw = label_enc.transform(df["label"].values)

        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        sequences, labels = [], []
        for i in range(len(X_scaled) - seq_len):
            sequences.append(X_scaled[i: i + seq_len])
            labels.append(Y_raw[i + seq_len - 1])

        return (
            np.array(sequences),
            np.array(labels),
            scaler,
            label_enc,
            feature_cols,
        )
    except Exception as e:
        print(f"Exception | prepare_data: {e} Line: {sys.exc_info()[2].tb_lineno}")


# Keras Custom Attention Layer 
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Nauceni vektor paznje
        self.attention = layers.Dense(1, use_bias=False)
        super().build(input_shape)

    def call(self, lstm_out):
        """
        lstm_out: (batch, seq_len, hidden_size)
        Vraca:
            context: (batch, hidden_size)
            weights: (batch, seq_len)
        """
        # Skorovi paznje za svaki timestep
        scores = self.attention(lstm_out)                    # (batch, seq_len, 1)
        weights = tf.nn.softmax(scores, axis=1)              # (batch, seq_len, 1)

        # Tezinski zbir hidden stateova
        context = tf.reduce_sum(weights * lstm_out, axis=1)  # (batch, hidden_size)
        weights = tf.squeeze(weights, axis=-1)               # (batch, seq_len)

        return context, weights


# Keras Custom Model
class DDoSLSTMAttention(models.Model):
    def __init__(self, hidden_size: int, num_layers: int, num_classes: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        
        self.lstm_layers = []
        for i in range(num_layers):
            # U Keras-u moramo vratiti sekvence da bi Attention radio pravilno, i za prosledjivanje narednom LSTM sloju
            self.lstm_layers.append(layers.LSTM(hidden_size, return_sequences=True))
            if i < num_layers - 1 and dropout > 0.0:
                self.lstm_layers.append(layers.Dropout(dropout))
                
        self.attention = AttentionLayer()
        
        self.classifier = models.Sequential([
            layers.Dropout(dropout),
            layers.Dense(hidden_size // 2, activation='relu'),
            layers.Dense(num_classes) # Logits (bez softmax aktivacije, softmax koristimo pri evaluaciji)
        ])

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.lstm_layers:
            x = layer(x, training=training)
            
        context, weights = self.attention(x)
        logits = self.classifier(context, training=training)
        return logits, weights

    # Override metoda da bi keras .fit() ignorisao 'weights' tokom racunanja loss-a
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits, _ = self(x, training=True)
            loss = self.compiled_loss(y, logits, regularization_losses=self.losses)
            
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        logits, _ = self(x, training=False)
        loss = self.compiled_loss(y, logits, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}


# Pomocne funkcije za plotovanje (Ostavljene netaknute)
def plot_confusion_matrix(y_true, y_pred, label_names: list[str]):
    try:
        os.makedirs("graphs", exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        for ax, data, title, fmt in zip(
            axes, [cm, cm_norm],
            ["Confusion Matrix (absolute numbers)", "Confusion Matrix (normalized)"],
            ["d", ".2f"],
        ):
            sns.heatmap(
                data, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=label_names, yticklabels=label_names,
                ax=ax, linewidths=0.5,
            )
            ax.set_title(title, fontsize=13, pad=12)
            ax.set_xlabel("Predicted", fontsize=11)
            ax.set_ylabel("Actual",    fontsize=11)
            ax.tick_params(axis="x", rotation=45)
            ax.tick_params(axis="y", rotation=0)

        plt.tight_layout()
        plt.savefig("graphs/confusion_matrix_tf.png", dpi=150)
        plt.show()
        print("Saved confusion matrix graphs/confusion_matrix_tf.png")
    except Exception as e:
        print(f'Exception | plot_confusion_matrix: {e}')

def plot_roc_curves(y_bin, all_probs, label_names: list[str]):
    try:
        os.makedirs("graphs", exist_ok=True)
        n_classes = len(label_names)
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

        plt.figure(figsize=(10, 7))
        for i, (name, color) in enumerate(zip(label_names, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
            auc = roc_auc_score(y_bin[:, i], all_probs[:, i])
            plt.plot(fpr, tpr, color=color, lw=1.8, label=f"{name}  (AUC = {auc:.3f})")

        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate",  fontsize=12)
        plt.title("ROC curves by class (one-vs-rest)", fontsize=13)
        plt.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        plt.savefig("graphs/roc_curves_tf.png", dpi=150)
        plt.show()
        print("Saved ROC curve graphs/roc_curves_tf.png")
    except Exception as e:
        print(f'Exception | plot_roc_curves: {e}')


# Kros-validacija i trening
def cross_validate(csv_path: str, save_path: str = "ddos_lstm_attention.keras"):
    try:
        sequences, labels, scaler, le, feature_cols = prepare_data(csv_path)
        num_features = len(feature_cols)
        num_classes  = len(le.classes_)

        tss = TimeSeriesSplit(n_splits=N_SPLITS)
        fold_metrics = []
        best_overall_loss = float("inf")

        for fold, (train_idx, val_idx) in enumerate(tss.split(sequences), start=1):
            print(f"\n{'='*60}")
            print(f"  Fold {fold}/{N_SPLITS} | Train: {len(train_idx):,}  Val: {len(val_idx):,}")
            print(f"{'='*60}")

            X_train, y_train = sequences[train_idx], labels[train_idx]
            X_val, y_val = sequences[val_idx], labels[val_idx]

            # Koriscenje tf.data API-ja za optimizovan data pipeline
            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

            # Inicijalizacija modela (nova instanca za svaki fold)
            model = DDoSLSTMAttention(
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                num_classes=num_classes,
                dropout=DROPOUT
            )

            # Gradient clipping je integrisan u optimizer
            optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
            
            # from_logits=True posto nas poslednji Dense sloj nema aktivaciju
            model.compile(
                optimizer=optimizer,
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )

            lr_scheduler = callbacks.ReduceLROnPlateau(
                monitor='val_loss', patience=3, factor=0.5, verbose=1
            )

            # Treniranje folda
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS,
                callbacks=[lr_scheduler],
                verbose=1
            )

            # Manualna evaluacija na validacionom setu za racunanje napredne metrike
            val_logits, val_weights = model.predict(val_ds)
            val_probs = tf.nn.softmax(val_logits, axis=-1).numpy()
            val_preds = np.argmax(val_probs, axis=-1)

            v_loss = history.history['val_loss'][-1]
            v_acc = history.history['val_accuracy'][-1]
            mcc = matthews_corrcoef(y_val, val_preds)

            metrics = {
                "fold":      fold,
                "val_loss":  v_loss,
                "val_acc":   v_acc,
                "mcc":       mcc,
                "preds":     val_preds,
                "targets":   y_val,
                "probs":     val_probs,
            }
            fold_metrics.append(metrics)

            if v_loss < best_overall_loss:
                best_overall_loss = v_loss
                # cuvamo model u nativnom Keras formatu
                model.save_weights(save_path)
                print(f"  New best model saved (fold {fold}) -> {save_path}")

        # Sumarni rezultati
        print(f"\n{'='*60}")
        print("  Cross-validation summary")
        print(f"{'='*60}")
        print(f"  {'Fold':<8} {'Val Loss':<12} {'Val Acc':<12} {'MCC'}")
        print(f"  {'-'*48}")

        for m in fold_metrics:
            print(f"  {m['fold']:<8} {m['val_loss']:<12.4f} {m['val_acc']:<12.4f} {m['mcc']:.4f}")

        avg_loss = np.mean([m["val_loss"] for m in fold_metrics])
        avg_acc  = np.mean([m["val_acc"]  for m in fold_metrics])
        avg_mcc  = np.mean([m["mcc"]      for m in fold_metrics])
        std_acc  = np.std( [m["val_acc"]  for m in fold_metrics])

        print(f"  {'-'*48}")
        print(f"  {'Avg':<8} {avg_loss:<12.4f} {avg_acc:<12.4f} {avg_mcc:.4f}")
        print(f"  {'Std':<8} {'':12} {std_acc:<12.4f}")
        print(f"{'='*60}\n")

        # Najbolji fold
        best_fold = min(fold_metrics, key=lambda m: m["val_loss"])
        best_targets = np.array(best_fold["targets"])
        best_preds   = np.array(best_fold["preds"])
        best_probs   = best_fold["probs"]

        print(f"Classification report (best fold {best_fold['fold']}):")
        print(classification_report(
            best_targets, best_preds,
            labels=list(range(num_classes)),
            target_names=LABELS,
            digits=4,
            zero_division=0,
        ))
        print(f"Matthews Correlation Coefficient (best fold {best_fold['fold']}): {best_fold['mcc']:.4f}\n")

        # Plotovanje
        plot_confusion_matrix(best_targets, best_preds, LABELS)
        y_bin = label_binarize(best_targets, classes=list(range(num_classes)))
        plot_roc_curves(y_bin, best_probs, LABELS)

        # Export metrika u JSON 
        print("Exporting evaluation metrics to JSON...")
        os.makedirs("results", exist_ok=True)
        json_path = export_metrics_to_json(
            y_true=best_targets,
            y_pred=best_preds,
            y_proba=best_probs,
            class_labels=LABELS,
            output_dir="results/",
            filename="eval_metrics_tf.json",
        )
        print(f"Metrics saved -> {json_path}")
        print("Training finished!")
        
        return fold_metrics

    except Exception as e:
        print(f"Exception | cross_validate: {e} Line: {sys.exc_info()[2].tb_lineno}")

if __name__ == "__main__":
    path = input('Insert csv file path: ').strip()
    cross_validate(path)