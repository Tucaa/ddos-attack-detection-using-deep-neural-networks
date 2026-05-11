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


# Opsezi hyperparametara za pretragu 

HYPERPARAMETER_SPACE = {
    "hidden_size":   [64, 128, 256, 512],
    "num_layers":    [1, 2, 3],
    "dropout":       (0.1, 0.5),       # Kontinualni opseg
    "learning_rate": (1e-4, 1e-2),     # Log opseg
    "batch_size":    [32, 64, 128],
    "seq_len":       [20, 30, 50, 75],
}

N_TRIALS    = 50    # Broj Optuna trial-ova
N_SPLITS    = 3     # Manji broj foldova zbog brzine tuninga
EPOCHS      = 10    # Manji broj epocha tokom tuninga
STUDY_NAME  = "ddos_lstm_attention_tuning"


# Optuna objective funkcija 

def objective(trial: optuna.Trial, sequences: np.ndarray, labels: np.ndarray, num_features: int, num_classes: int, device) -> float:
    """
    Optuna poziva ovu funkciju za svaki trial.
    Vraca prosecni val_loss kroz foldove — Optuna minimizuje ovu vrednost.
    """
    try:
        #Predlozi hyperparametre za ovaj trial
        hidden_size   = trial.suggest_categorical("hidden_size",   HYPERPARAMETER_SPACE["hidden_size"])
        num_layers    = trial.suggest_categorical("num_layers",    HYPERPARAMETER_SPACE["num_layers"])
        dropout       = trial.suggest_float("dropout",             *HYPERPARAMETER_SPACE["dropout"])
        learning_rate = trial.suggest_float("learning_rate",       *HYPERPARAMETER_SPACE["learning_rate"], log=True)
        batch_size    = trial.suggest_categorical("batch_size",    HYPERPARAMETER_SPACE["batch_size"])
        seq_len       = trial.suggest_categorical("seq_len",       HYPERPARAMETER_SPACE["seq_len"])

        # Sekvence se prave na osnovu predlozenog seq_len
        seqs, lbls = _make_sequences(sequences, labels, seq_len)
        dataset    = DDoSDataset(seqs, lbls)
        tss        = TimeSeriesSplit(n_splits=N_SPLITS)
        fold_losses = []

        for fold, (train_idx, val_idx) in enumerate(tss.split(seqs)):
            train_loader = DataLoader(
                Subset(dataset, train_idx),
                batch_size=batch_size,
                shuffle=False,
            )
            val_loader = DataLoader(
                Subset(dataset, val_idx),
                batch_size=batch_size,
                shuffle=False,
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

            best_fold_loss = float("inf")

            for epoch in range(EPOCHS):
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

                if val_loss < best_fold_loss:
                    best_fold_loss = val_loss

                # Pruning — Optuna odustaje od obecavajucih trial-ova rano
                trial.report(val_loss, step=fold * EPOCHS + epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            fold_losses.append(best_fold_loss)

        avg_loss = float(np.mean(fold_losses))
        return avg_loss

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Exception | objective: {e} Line: {sys.exc_info()[2].tb_lineno}")
        return float("inf")


# Pomocne funkcije

def _make_sequences(raw_sequences: np.ndarray, labels: np.ndarray, seq_len: int):
    """
    Pravi sekvence kliznim prozorom za dati seq_len.
    Koristi se unutar objective f-je jer se seq_len tunira.
    """
    # raw_sequences su vec maksimalne duzine — uzimamo prvih seq_len kolona
    if raw_sequences.shape[1] >= seq_len:
        seqs = raw_sequences[:, :seq_len, :]
        lbls = labels
    else:
        seqs = raw_sequences
        lbls = labels
    return seqs, lbls


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

def run_hyperparameter_tuning(csv_path: str, results_path: str = "best_hyperparams.json"):
    """
    Pokrece Optuna hyperparameter tuning za LSTM+Attention model.
    Na kraju cuva najbolje hyperparametre u JSON fajl.
    Koristi TPE sampler — pametniji od random pretrage.
    Koristi MedianPruner — zaustavlja lose trial-ove rano.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using: {device}")
        print(f"Starting hyperparameter tuning: {N_TRIALS} trials, {N_SPLITS} folds, {EPOCHS} epochs/fold\n")

        # Ucitavanje podataka jednom — deli se kroz sve trial-ove
        sequences, labels, scaler, le, feature_cols = prepare_data(csv_path, seq_len=max(HYPERPARAMETER_SPACE["seq_len"]))
        num_features = len(feature_cols)
        num_classes  = len(le.classes_)

        print(f"Dataset loaded: {len(sequences):,} sequences | Features: {num_features} | Classes: {num_classes}\n")

        # Kreiranje Optuna study-a
        study = optuna.create_study(
            study_name=STUDY_NAME,
            direction="minimize",        # Minimizujemo val_loss
            sampler=TPESampler(seed=42), # TPE — uci iz prethodnih trial-ova
            pruner=MedianPruner(
                n_startup_trials=10,     # Prvih 10 trial-ova bez pruning-a
                n_warmup_steps=5,        # Prvih 5 epocha bez pruning-a
            ),
        )

        # Callback za stampanje posle svakog trial-a
        def trial_callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                _print_trial_summary(trial)
                if trial.number == study.best_trial.number:
                    print(f"  *** New best trial! ***")

        study.optimize(
            lambda trial: objective(trial, sequences, labels, num_features, num_classes, device),
            n_trials=N_TRIALS,
            callbacks=[trial_callback],
            show_progress_bar=True,
        )

        # ── Rezultati ─────────────────────────────────────────────────
        best = study.best_trial
        print(f"\n{'='*60}")
        print(f"  Tuning finished | Best loss: {best.value:.4f}")
        print(f"{'='*60}")
        print(f"  Best hyperparameters:")
        for k, v in best.params.items():
            print(f"    {k:<20} {v}")
        print(f"{'='*60}\n")

        # Cuvanje najboljeg u JSON
        best_params = {
            **best.params,
            "best_val_loss": best.value,
            "n_trials":      N_TRIALS,
            "study_name":    STUDY_NAME,
        }
        with open(results_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"Best hyperparameters saved -> {results_path}")

        # Optuna vizualizacije
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