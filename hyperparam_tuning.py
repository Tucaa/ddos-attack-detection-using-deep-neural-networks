import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import sys
import json
from torch_nn import *


# Opsezi hyperparametara za pretragu (po potrebi rekonfigurisati)
HYPERPARAMETER_SPACE = {
    "hidden_size":   [64, 128, 256],
    "num_layers":    [1, 2],
    "dropout":       (0.1, 0.4),
    "learning_rate": (1e-4, 1e-2),
    "seq_len":       [20, 30, 50],
}

TUNING_BATCH_SIZE = 256
TUNING_DATA_RATIO  = 0.3    # Koristi samo 30% dataseta tokom tuninga
TUNING_EPOCHS      = 5     
N_TRIALS   = 30   # Smanjeno sa 50 — TPE konvergira brze nego random search
STUDY_NAME = "ddos_lstm_attention_tuning"


# Optuna objective funkcija 
def objective_fast(trial: optuna.Trial,precomputed: dict, num_features: int, num_classes: int, device) -> float:
    """
    Brza verzija objective funkcije.
    Koristi jedan train/val split umesto K-fold i pre-computed sekvence.
    K-fold cross-validacija se radi samo sa najboljim hyperparametrima na kraju.
    """
    try:
        hidden_size   = trial.suggest_categorical("hidden_size",   HYPERPARAMETER_SPACE["hidden_size"])
        num_layers    = trial.suggest_categorical("num_layers",     HYPERPARAMETER_SPACE["num_layers"])
        dropout       = trial.suggest_float("dropout",              *HYPERPARAMETER_SPACE["dropout"])
        learning_rate = trial.suggest_float("learning_rate",        *HYPERPARAMETER_SPACE["learning_rate"], log=True)
        batch_size    = TUNING_BATCH_SIZE   # Fiksan tokom tuninga radi brzine
        seq_len       = trial.suggest_categorical("seq_len",        HYPERPARAMETER_SPACE["seq_len"])

        # Sekvence su vec preracunate
        sequences, labels = precomputed[seq_len]

        # Jedan temporalni split umesto K-fold
        split     = int(len(sequences) * 0.8)
        # X_train   = torch.tensor(sequences[:split], dtype=torch.float32)
        # y_train   = torch.tensor(labels[:split],    dtype=torch.long)
        # X_val     = torch.tensor(sequences[split:], dtype=torch.float32)
        # y_val     = torch.tensor(labels[split:],    dtype=torch.long)

        train_loader = DataLoader(
            DDoSDataset(sequences[:split], labels[:split]),
            batch_size=batch_size, shuffle=False,
        )
        val_loader = DataLoader(
            DDoSDataset(sequences[split:], labels[split:]),
            batch_size=batch_size, shuffle=False,
        )

        model = DDoSLSTMAttention(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")

        for epoch in range(TUNING_EPOCHS):
            # Trening
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits, _ = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Validacija
            val_loss = _evaluate_loss(model, val_loader, criterion, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Pruning — zaustavlja lose trial-ove rano
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_val_loss

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Exception | objective_fast: {e} Line: {sys.exc_info()[2].tb_lineno}")
        return float("inf")


# Pomocne funkcije
def precompute_all_sequences(csv_path: str, seq_lens: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Ucitava CSV jednom i pravi sekvence za svaki seq_len unapred.
    Cuva u recnik {seq_len: (sequences, labels)}.
    Izbegava ponavljanje ovog posla unutar svake Optuna iteracije.
    """
    print("Pre-computing sequences for all seq_len values...")

    # Ucitaj sirove podatke jednom
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in EXCLUDED_COLS]
    X_raw = df[feature_cols].values.astype(np.float32)

    label_enc = LabelEncoder()
    label_enc.classes_ = np.array(LABELS)
    Y_raw = label_enc.transform(df["label"].values)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Uzmemo samo TUNING_DATA_RATIO % podataka za tuning
    n_samples  = int(len(X_scaled) * TUNING_DATA_RATIO)
    X_subset   = X_scaled[:n_samples]
    Y_subset   = Y_raw[:n_samples]

    precomputed = {}
    for seq_len in seq_lens:
        seqs, lbls = [], []
        for i in range(len(X_subset) - seq_len):
            seqs.append(X_subset[i: i + seq_len])
            lbls.append(Y_subset[i + seq_len - 1])
        precomputed[seq_len] = (np.array(seqs), np.array(lbls))
        print(f"  seq_len={seq_len:<4} -> {len(seqs):,} sequences")

    print(f"Pre-computation done. Using {n_samples:,}/{len(X_scaled):,} samples for tuning.\n")
    return precomputed, scaler, label_enc, len(feature_cols)


@torch.no_grad()
def _evaluate_loss(model, loader, criterion, device) -> float:
    """
    Racuna prosecni loss na validacionom setu.
    """
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits, _ = model(X_batch)
        total_loss += criterion(logits, y_batch).item() * len(y_batch)
    return total_loss / len(loader.dataset)


def _print_trial_summary(trial: optuna.Trial):
    """
    Stampa rezultate jednog trial-a.
    """
    print(
        f"  Trial {trial.number:>3} | "
        f"Loss: {trial.value:.4f} | "
        f"Params: hidden={trial.params.get('hidden_size')} "
        f"layers={trial.params.get('num_layers')} "
        f"dropout={trial.params.get('dropout'):.3f} "
        f"lr={trial.params.get('learning_rate'):.2e} "
        f"batch={trial.params.get('batch_size')} "
        f"seq={trial.params.get('seq_len')}"
    )


# Glavna funkcija za tuning 
def run_hyperparameter_tuning(csv_path: str,results_path: str = "best_hyperparams.json",):
    """
    Dvofazni tuning:
    Faza 1 - Brzi tuning na 30% podataka sa jednim foldom (Optuna)
    Faza 2 - Cross-validacija samo sa najboljim hyperparametrima
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using: {device}")

        # Pre-compute jednom za sve trial-ove
        seq_lens = HYPERPARAMETER_SPACE["seq_len"]
        precomputed, scaler, le, num_features = precompute_all_sequences(csv_path, seq_lens)
        num_classes = len(le.classes_)

        # Faza 1: Brzi Optuna tuning
        print(f"Phase 1: Fast tuning ({N_TRIALS} trials, {TUNING_EPOCHS} epochs each)\n")

        study = optuna.create_study(
            study_name=STUDY_NAME,
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=3),
        )

        def trial_callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                _print_trial_summary(trial)
                if trial.number == study.best_trial.number:
                    print(f"  *** New best! ***")
            elif trial.state == optuna.trial.TrialState.PRUNED:
                print(f"  Trial {trial.number:>3} | Pruned")

        study.optimize(
            lambda trial: objective_fast(trial, precomputed, num_features, num_classes, device),
            n_trials=N_TRIALS,
            callbacks=[trial_callback],
            show_progress_bar=True,
        )

        best = study.best_trial
        print(f"\n{'='*60}")
        print(f"  Phase 1 done | Best loss: {best.value:.4f}")
        print(f"  Best params: {best.params}")
        print(f"{'='*60}\n")

        # Faza 2: Cross-validacija sa najboljim parametrima 
        print("Phase 2: Full cross-validation with best hyperparameters\n")

        global HIDDEN_SIZE, NUM_LAYERS, DROPOUT, LEARNING_RATE, BATCH_SIZE, SEQUENCE_LEN, EPOCHS
        HIDDEN_SIZE   = best.params["hidden_size"]
        NUM_LAYERS    = best.params["num_layers"]
        DROPOUT       = best.params["dropout"]
        LEARNING_RATE = best.params["learning_rate"]
        BATCH_SIZE    = best.params["batch_size"] if "batch_size" in best.params else 64
        SEQUENCE_LEN  = best.params["seq_len"]
        EPOCHS        = 20  # Vracamo pun broj epocha za finalni trening

        cv_results = cross_validate(csv_path, save_path="ddos_best_model.pt")

        # Cuvanje
        best_params = {
            **best.params,
            "best_tuning_loss": best.value,
            "n_trials":         N_TRIALS,
        }
        with open(results_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"\nBest hyperparameters saved -> {results_path}")

        _plot_optuna_results(study)
        return best_params

    except Exception as e:
        print(f"Exception | run_hyperparameter_tuning: {e} Line: {sys.exc_info()[2].tb_lineno}")


def _plot_optuna_results(study: optuna.Study):
    """
    Crta Optuna vizualizacije:
    - Historia trial-ova (loss kroz vreme)
    - Vaznost hyperparametara
    - Korelacija parametara
    """
    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
        )
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        plt.sca(axes[0])
        plot_optimization_history(study)
        axes[0].set_title("Optimization history", fontsize=12)

        plt.sca(axes[1])
        plot_param_importances(study)
        axes[1].set_title("Hyperparameter importance", fontsize=12)

        plt.sca(axes[2])
        plot_parallel_coordinate(study)
        axes[2].set_title("Parallel coordinates", fontsize=12)

        plt.tight_layout()
        plt.savefig("optuna_results.png", dpi=150)
        print("Optuna plots saved -> optuna_results.png")
        plt.show()

    except Exception as e:
        print(f"Exception | _plot_optuna_results: {e} Line: {sys.exc_info()[2].tb_lineno}")


# Trening finalnog modela sa najboljim hyperparametrima

def train_best_model(csv_path: str, params_path: str = "best_hyperparams.json", save_path: str = "ddos_best_model.pt"):
    """
    Ucitava najbolje hyperparametre iz JSON fajla i trenira finalni model
    sa cross-validacijom na celom datasetu.
    """
    try:
        with open(params_path) as f:
            params = json.load(f)

        print(f"Loaded best hyperparameters from: {params_path}")
        for k, v in params.items():
            print(f"  {k:<20} {v}")
        print()

        # Pokretanje cross-validacije sa najboljim parametrima
        # Privremeno override-ujemo globalne konstante
        global HIDDEN_SIZE, NUM_LAYERS, DROPOUT, LEARNING_RATE, BATCH_SIZE, SEQUENCE_LEN
        HIDDEN_SIZE   = params["hidden_size"]
        NUM_LAYERS    = params["num_layers"]
        DROPOUT       = params["dropout"]
        LEARNING_RATE = params["learning_rate"]
        BATCH_SIZE    = params["batch_size"]
        SEQUENCE_LEN  = params["seq_len"]

        cross_validate(csv_path, save_path=save_path)

    except Exception as e:
        print(f"Exception | train_best_model: {e} Line: {sys.exc_info()[2].tb_lineno}")


if __name__ == "__main__":
    csv_path = input("CSV file path: ").strip()

    print("Select mode:")
    print("  1 - Run hyperparameter tuning")
    print("  2 - Train final model with best hyperparameters")
    choice = input("Choice (1/2): ").strip()

    if choice == "1":
        run_hyperparameter_tuning(csv_path)
    elif choice == "2":
        params_path = input("Path to best_hyperparams.json: ").strip() or "best_hyperparams.json"
        train_best_model(csv_path, params_path)
    else:
        print("Invalid choice.")