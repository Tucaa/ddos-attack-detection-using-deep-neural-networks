
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def export_metrics_to_json(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, class_labels: list[str], output_dir: str = ".", filename: str = "eval_metrics.json") -> str:
    """
    Racuna sve metrike evaluacije i cuva ih u JSON fajl.

    Parametri:
        y_true       — stvarne oznake (integer indeksi klasa)
        y_pred       — predvidjene oznake (integer indeksi klasa)
        y_proba      — verovatnoca po klasi (shape: n_samples x n_classes)
        class_labels — lista naziva klasa u ispravnom redosledu
        output_dir   — direktorijum za cuvanje JSON fajla
        filename     — naziv izlaznog fajla

    Vraca:
        Apsolutnu putanju do sacuvanog JSON fajla.
    """
    try:
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_labels,
            output_dict=True,
            zero_division=0,
        )
        # Cuvamo samo per-class stavke (iskljucujemo accuracy, macro avg, weighted avg)
        per_class_report = {
            cls: metrics
            for cls, metrics in report.items()
            if isinstance(metrics, dict)
        }

        cm = confusion_matrix(y_true, y_pred)

        mcc = float(matthews_corrcoef(y_true, y_pred))

        roc_auc = _compute_roc_auc_per_class(y_true, y_proba, class_labels)

        # Agregiran rezultat
        metrics_payload = {
            "timestamp":             datetime.now().isoformat(),
            "class_labels":          class_labels,
            "mcc_score":             mcc,
            "classification_report": per_class_report,
            "confusion_matrix":      cm.tolist(),
            "roc_auc_scores":        roc_auc,
            # Agregirane vrednosti za brzi pregled
            "summary": {
                "macro_f1":       report.get("macro avg", {}).get("f1-score", 0.0),
                "weighted_f1":    report.get("weighted avg", {}).get("f1-score", 0.0),
                "macro_precision": report.get("macro avg", {}).get("precision", 0.0),
                "macro_recall":   report.get("macro avg", {}).get("recall", 0.0),
                "total_samples":  int(report.get("macro avg", {}).get("support", 0)),
            },
        }


        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)

        logger.info(f"Metrics saved to: {output_path}")
        return str(output_path.resolve())

    except Exception as e:
        logger.error(f"Failed to export metrics: {e} | Line: {sys.exc_info()[2].tb_lineno}")
        raise


def _compute_roc_auc_per_class(y_true: np.ndarray, y_proba: np.ndarray, class_labels: list[str],) -> dict[str, float]:
    """
    Racuna ROC-AUC za svaku klasu metodom one-vs-rest.
    Vraca 0.0 za klasu koja nije prisutna u y_true (ne moze se izracunati).
    """
    roc_auc: dict[str, float] = {}

    for i, label in enumerate(class_labels):
        if i >= y_proba.shape[1]:
            roc_auc[label] = 0.0
            continue

        y_true_binary = (y_true == i).astype(int)

        # Ako klasa nije prisutna u skupu, AUC se ne moze izracunati
        if y_true_binary.sum() == 0:
            logger.warning(f"Class '{label}' not present in y_true — ROC-AUC set to 0.0")
            roc_auc[label] = 0.0
            continue

        try:
            roc_auc[label] = float(roc_auc_score(y_true_binary, y_proba[:, i]))
        except Exception as e:
            logger.warning(f"ROC-AUC failed for class '{label}': {e}")
            roc_auc[label] = 0.0

    return roc_auc


def load_metrics_from_json(path: str) -> Optional[dict]:
    """
    Ucitava prethodno sacuvane metrike iz JSON fajla.
    Vraca None ako fajl ne postoji ili je ostecen.
    """
    p = Path(path)
    if not p.exists():
        logger.warning(f"Metrics file not found: {path}")
        return None

    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load metrics from '{path}': {e}")
        return None