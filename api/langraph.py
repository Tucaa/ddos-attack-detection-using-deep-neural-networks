import json
import logging
from pathlib import Path
from typing import TypedDict, Optional
import shutil
import asyncio
import subprocess

from langgraph.graph import StateGraph, END

from ollama_client import ollama_generate, _extract_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

METRICS_PATH      = Path("results/test_metrics.json")
PREV_METRICS_PATH = Path("results/test_metrics_prev.json")

# Pragovi za decision logiku — čisti Python, bez LLM-a
DECISION_RETRAIN_MCC    = 0.75   # MCC ispod ovoga => obavezno RETRAIN
DECISION_SCAN_MCC       = 0.85   # MCC ispod ovoga => SCAN_FIRST
# Iznad DECISION_SCAN_MCC i bez regresije => SUGGEST_ONLY

# Mapiranje: tip napada => fajlovi relevantni za skeniranje
# Uvek se čitaju config.py i hyperparam.py
ALWAYS_SCAN = ["hyperparam.py", "config.py"]

# Fajlovi relevantni po klasi slabih performansi
CLASS_TO_FILES: dict[str, list[str]] = {
    "udp_flood_large":       ["attacks.py", "dataset_generator.py"],
    "udp_flood_mixed":       ["attacks.py", "dataset_generator.py"],
    "dns_amplification":     ["attacks.py", "dataset_generator.py"],
    "ntp_amplification":     ["attacks.py", "dataset_generator.py"],
    "syn_flood":             ["attacks.py", "windowing.py"],
    "ack_flood":             ["attacks.py", "windowing.py"],
    "icmp_flood":            ["attacks.py", "dataset_generator.py"],
    "subnet_carpet_bombing": ["attacks.py", "dataset_generator.py"],
    "normal":                ["dataset_generator.py", "normal.py"],
}

# Maksimalan broj karaktera po fajlu koji se prosledjuje LLM-u
# (sprecava prekoracenje kontekst prozora Ollame)
SCAN_MAX_CHARS_PER_FILE = 3000

# Folderi koji se preskačaju pri rekurzivnom skeniranju
SCAN_SKIP_DIRS: set[str] = {"__pycache__", "venv", ".venv", ".git", ".idea"}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class MetricsState(TypedDict):
    # --- Ulaz (trenutni run) ---
    classification_report: dict   # {klasa: {precision, recall, f1-score, support}}
    confusion_matrix: list        # 2D lista integera (num_classes x num_classes)
    mcc_score: float
    roc_auc_scores: dict          # {klasa: float}
    class_labels: list            # lista naziva klasa u ispravnom redosledu

    # --- Poređenje sa prethodnim runom ---
    previous_metrics: dict        # sadrzaj test_metrics_prev.json, {} ako ne postoji
    metrics_delta: dict           # {mcc_delta, macro_f1_delta, per_class_f1_delta: {klasa: delta}}
    regression_detected: bool     # True ako je bilo koji indikator gori nego pre

    # --- Medjurezultati analize ---
    per_class_analysis: str
    confusion_analysis: str

    # --- Izlaz analize ---
    weak_classes: list            # nazivi klasa sa losim performansama
    recommendations: dict         # {data_generation: [], model: [], training: []}
    summary: str

    # --- Decision i akcija ---
    decision: str                 # "SUGGEST_ONLY" | "SCAN_FIRST" | "RETRAIN"
    scanned_files: dict           # {filename: sadrzaj} — puni node_scan_codebase
    proposed_hyperparams: dict    # predlozene vrednosti iz HYPERPARAMETER_SPACE
    human_confirmed: bool         # da li je korisnik potvrdio retrain
    retrain_triggered: bool       # da li je retrain pokrenut
    retrain_command: str          # komanda koja je izvrsena (za logovanje)



def _format_per_class_metrics(
    report: dict,
    roc_auc: dict,
    mcc: float,
) -> str:
    """Formatira per-class metrike u citljiv string za LLM prompt."""
    lines = [f"Overall MCC: {mcc:.4f}\n"]
    for cls, m in report.items():
        if not isinstance(m, dict):
            continue
        lines.append(
            f"  {cls}:\n"
            f"    precision={m.get('precision', 0):.3f}  "
            f"recall={m.get('recall', 0):.3f}  "
            f"f1={m.get('f1-score', 0):.3f}  "
            f"support={int(m.get('support', 0))}  "
            f"roc_auc={roc_auc.get(cls, 0):.3f}"
        )
    return "\n".join(lines)


def _format_confusion_matrix(matrix: list, labels: list) -> str:
    """Formatira matricu konfuzije kao ASCII tabelu za LLM prompt."""
    col_width = max(len(l) for l in labels) + 2
    header = " " * col_width + "  ".join(f"{l:>{col_width}}" for l in labels)
    rows = []
    for i, row in enumerate(matrix):
        row_str = f"{labels[i]:>{col_width}}  " + "  ".join(
            f"{v:>{col_width}}" for v in row
        )
        rows.append(row_str)
    return header + "\n" + "\n".join(rows)


def _safe_parse_json(raw: str, fallback: dict) -> dict:
    """Pokusava da parsira JSON iz LLM odgovora, vraca fallback ako ne uspe."""
    clean = _extract_json(raw)
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing failed: {e} | Raw snippet: {raw[:200]}")
        return fallback


# ---------------------------------------------------------------------------
# Helper: računanje delte između dva run-a
# ---------------------------------------------------------------------------

def _compute_delta(current: dict, previous: dict) -> tuple[dict, bool]:
    """
    Poredi trenutne metrike sa prethodnim runom.

    Vraca:
        delta (dict) — razlike za MCC, macro F1 i per-class F1
        regression_detected (bool) — True ako je MCC ili macro F1 opao,
                                     ili ako je neka klasa izgubila >= 0.03 F1
    """
    if not previous:
        # Nema prethodnog run-a — nema sta da se poredi
        return {
            "mcc_delta":        None,
            "macro_f1_delta":   None,
            "per_class_f1_delta": {},
            "note": "No previous run found — first baseline.",
        }, False

    # MCC delta
    prev_mcc  = previous.get("mcc_score", 0.0)
    curr_mcc  = current["mcc_score"]
    mcc_delta = round(curr_mcc - prev_mcc, 4)

    # Macro F1 delta (iz summary polja ako postoji)
    prev_f1  = previous.get("summary", {}).get("macro_f1", 0.0)
    curr_f1  = current.get("summary", {}).get("macro_f1", 0.0)
    f1_delta = round(curr_f1 - prev_f1, 4) if curr_f1 and prev_f1 else None

    # Per-class F1 delta
    prev_report = previous.get("classification_report", {})
    curr_report = current["classification_report"]
    per_class_delta: dict[str, float] = {}

    for cls, curr_m in curr_report.items():
        if not isinstance(curr_m, dict):
            continue
        prev_m = prev_report.get(cls, {})
        if isinstance(prev_m, dict):
            diff = round(
                curr_m.get("f1-score", 0.0) - prev_m.get("f1-score", 0.0), 4
            )
            per_class_delta[cls] = diff

    # Regresija ako je MCC opao, F1 opao, ili neka klasa izgubila >= 0.03 F1
    regression = (
        mcc_delta < 0
        or (f1_delta is not None and f1_delta < 0)
        or any(v < -0.03 for v in per_class_delta.values())
    )

    delta = {
        "mcc_delta":          mcc_delta,
        "macro_f1_delta":     f1_delta,
        "per_class_f1_delta": per_class_delta,
    }

    return delta, regression


# ---------------------------------------------------------------------------
# Node 0: Učitavanje i poređenje metrika
# ---------------------------------------------------------------------------

async def node_load_and_compare_metrics(state: MetricsState) -> dict:
    """
    Učitava prethodni run iz test_metrics_prev.json (ako postoji) i
    računa delta vrednosti u odnosu na trenutne metrike.

    Popunjava: previous_metrics, metrics_delta, regression_detected

    Ne poziva Ollama — čista Python logika.
    Mora biti prvi čvor u grafu jer ostali čvorovi koriste delta kontekst.
    """
    logger.info("[LangGraph] node_load_and_compare_metrics: start")

    # Pokušaj učitavanja prethodnih metrika
    previous: dict = {}
    if PREV_METRICS_PATH.exists():
        try:
            with open(PREV_METRICS_PATH, "r") as f:
                previous = json.load(f)
            logger.info(
                f"Previous metrics loaded: MCC={previous.get('mcc_score', 'N/A')}, "
                f"timestamp={previous.get('timestamp', 'N/A')}"
            )
        except Exception as e:
            logger.warning(f"Could not load previous metrics: {e} — treating as first run.")
            previous = {}
    else:
        logger.info("No previous metrics file found — this is treated as first baseline run.")

    # Računanje delte
    # Trenutne metrike su već u state-u (učitane u langraph_test.py)
    current_as_dict = {
        "mcc_score":             state["mcc_score"],
        "classification_report": state["classification_report"],
        "summary":               {},   # langraph_test prosleđuje summary ako postoji
    }
    delta, regression = _compute_delta(current_as_dict, previous)

    # Log najvažnijih promena
    if previous and delta.get("mcc_delta") is not None:
        sign = "▲" if delta["mcc_delta"] >= 0 else "▼"
        logger.info(
            f"  MCC delta: {sign} {abs(delta['mcc_delta']):.4f} "
            f"({'regression detected' if regression else 'no regression'})"
        )
        regressions = [
            f"{cls} ({v:+.3f})"
            for cls, v in delta.get("per_class_f1_delta", {}).items()
            if v < -0.03
        ]
        if regressions:
            logger.warning(f"Per-class regressions: {', '.join(regressions)}")

    logger.info("[LangGraph] node_load_and_compare_metrics: done")

    return {
        "previous_metrics":   previous,
        "metrics_delta":      delta,
        "regression_detected": regression,
    }


async def node_analyze_per_class(state: MetricsState) -> dict:
    """
    Analizira precision, recall, f1-score i ROC-AUC po klasi.
    Identifikuje slabe klase i razloge losih performansi.
    Popunjava: weak_classes, per_class_analysis
    """
    logger.info("[LangGraph] node_analyze_per_class: start")

    metrics_str = _format_per_class_metrics(
        state["classification_report"],
        state["roc_auc_scores"],
        state["mcc_score"],
    )

    # Dodajemo delta kontekst u prompt ako postoji poređenje sa prethodnim runom
    delta     = state.get("metrics_delta", {})
    has_delta = delta.get("mcc_delta") is not None

    delta_ctx = ""
    if has_delta:
        sign      = "improved" if delta["mcc_delta"] >= 0 else "regressed"
        delta_ctx = (
            f"\nCOMPARISON WITH PREVIOUS RUN:\n"
            f"  MCC delta: {delta['mcc_delta']:+.4f} ({sign})\n"
        )
        per_class = delta.get("per_class_f1_delta", {})
        if per_class:
            regressed = [(c, v) for c, v in per_class.items() if v < -0.03]
            improved  = [(c, v) for c, v in per_class.items() if v >  0.03]
            if regressed:
                delta_ctx += "  Classes with notable F1 regression (>0.03): " + \
                    ", ".join(f"{c} ({v:+.3f})" for c, v in regressed) + "\n"
            if improved:
                delta_ctx += "  Classes with notable F1 improvement (>0.03): " + \
                    ", ".join(f"{c} ({v:+.3f})" for c, v in improved) + "\n"
    else:
        delta_ctx = "\nCOMPARISON WITH PREVIOUS RUN: Not available (first run or no baseline).\n"

    prompt = f"""You are analyzing per-class performance of a DDoS traffic classification LSTM model.

    The model classifies network traffic windows into 9 classes:
    - normal traffic
    - 8 DDoS attack types: udp_flood_large, dns_amplification, subnet_carpet_bombing,
    syn_flood, icmp_flood, udp_flood_mixed, ntp_amplification, ack_flood

    Per-class metrics:
    {metrics_str}
    {delta_ctx}
    Thresholds for "weak": f1-score < 0.80 OR recall < 0.75 OR roc_auc < 0.85

    Respond ONLY with a valid JSON object (no markdown, no explanation outside JSON):
    {{
    "weak_classes": ["list of class names that fall below thresholds"],
    "analysis": "Technical paragraph (4-6 sentences) explaining WHY these specific classes underperform. Consider: similar traffic patterns between attack types, feature overlap (e.g., udp_flood_mixed vs udp_flood_large share packet size distributions), class imbalance, insufficient diversity in synthetic data generation, or transition period blending artifacts. If comparison data is available, note whether the situation improved or worsened versus the previous run."
    }}"""

    system = (
        "You are an ML engineer specializing in network traffic classification. "
        "You analyze model performance metrics and identify root causes of underperformance. "
        "You respond ONLY with valid JSON."
    )

    raw = await ollama_generate(prompt, system)
    result = _safe_parse_json(raw, fallback={"weak_classes": [], "analysis": raw[:500]})

    logger.info(
        f"[LangGraph] node_analyze_per_class: done | "
        f"weak_classes={result.get('weak_classes', [])}"
    )

    return {
        "weak_classes": result.get("weak_classes", []),
        "per_class_analysis": result.get("analysis", raw[:500]),
    }



async def node_analyze_confusion(state: MetricsState) -> dict:
    """
    Analizira matricu konfuzije i identifikuje obrasce gresaka klasifikacije.
    Popunjava: confusion_analysis
    """
    logger.info("[LangGraph] node_analyze_confusion: start")

    matrix_str = _format_confusion_matrix(
        state["confusion_matrix"],
        state["class_labels"],
    )

    # Prosledjujemo vec identifikovane slabe klase kao kontekst
    weak_ctx = (
        f"Previously identified weak classes: {', '.join(state['weak_classes'])}"
        if state["weak_classes"]
        else "No weak classes identified yet."
    )

    prompt = f"""You are analyzing a confusion matrix from a DDoS traffic classification model.

    Classes (in order): {", ".join(state["class_labels"])}

    Confusion Matrix (rows = actual class, columns = predicted class):
    {matrix_str}

    {weak_ctx}

    Identify the most significant misclassification patterns.

    Respond ONLY with a valid JSON object:
    {{
    "confusion_patterns": "Technical paragraph (4-6 sentences) describing which classes are most often confused with each other, likely reasons (similar traffic signatures, feature space overlap, insufficient discrimination in the LSTM's learned representations), and any systematic biases (e.g., model over-predicts 'normal').",
    "top_confusions": [
        {{"actual": "class_name", "predicted": "class_name", "count": 123}},
        {{"actual": "class_name", "predicted": "class_name", "count": 89}},
        {{"actual": "class_name", "predicted": "class_name", "count": 67}}
    ]
    }}"""

    system = (
        "You are an ML engineer specializing in network traffic classification. "
        "You respond ONLY with valid JSON."
    )

    raw = await ollama_generate(prompt, system)
    result = _safe_parse_json(
        raw,
        fallback={"confusion_patterns": raw[:500], "top_confusions": []},
    )

    logger.info("[LangGraph] node_analyze_confusion: done")

    return {
        "confusion_analysis": result.get("confusion_patterns", raw[:500]),
    }


async def node_synthesize_recommendations(state: MetricsState) -> dict:
    """
    Sintetizuje konkretne preporuke za poboljsanje na osnovu prethodnih analiza.
    Preporuke su podeljene u tri kategorije:
      - data_generation: izmene u sintetickom generatoru podataka
      - model: izmene u arhitekturi ili hiperparametrima LSTM-a
      - training: izmene u strategiji treniranja
    Popunjava: recommendations, summary
    """
    logger.info("[LangGraph] node_synthesize_recommendations: start")

    weak_str = (
        ", ".join(state["weak_classes"]) if state["weak_classes"] else "none identified"
    )

    delta     = state.get("metrics_delta", {})
    has_delta = delta.get("mcc_delta") is not None
    regression = state.get("regression_detected", False)

    trend_ctx = ""
    if has_delta:
        trend_ctx = (
            f"  Trend vs previous run: MCC {delta['mcc_delta']:+.4f}, "
            f"macro F1 {delta.get('macro_f1_delta', 0) or 0:+.4f} "
            f"({'REGRESSION DETECTED' if regression else 'no regression'})\n"
        )
    else:
        trend_ctx = "  Trend vs previous run: N/A (first run)\n"

    prompt = f"""You are synthesizing actionable improvement recommendations for a DDoS LSTM classifier.

    CONTEXT:
    - Model: 2-layer LSTM with attention, sliding window of 20 samples x 12 features
    - Features: packet_size, packets_per_second, bytes_per_second, src_port, dst_port,
    protocol, tcp_flags, ttl, inter_arrival_time, flow_duration, unique_src_ips, unique_dst_ports
    - Data: synthetic traffic with Gaussian noise, transition period blending, minority class oversampling
    - Training: CrossEntropyLoss with inverse-frequency class weights, Adam optimizer, ReduceLROnPlateau

    ANALYSIS RESULTS:
    - Overall MCC: {state["mcc_score"]:.4f}
    {trend_ctx}
    - Weak classes: {weak_str}
    - Per-class analysis: {state["per_class_analysis"]}
    - Confusion matrix analysis: {state["confusion_analysis"]}

    Generate concrete, specific, implementable recommendations.

    Respond ONLY with a valid JSON object:
    {{
    "data_generation": [
        "3-4 specific recommendations for improving the synthetic data generator (e.g., adjust noise std per feature, increase minority class oversampling ratio, modify transition blending window, add feature correlations specific to attack types)"
    ],
    "model": [
        "3-4 specific recommendations for LSTM architecture or hyperparameters (e.g., increase hidden_size, add bidirectional LSTM, adjust dropout rate, increase window_size, add batch normalization)"
    ],
    "training": [
        "2-3 specific recommendations for training strategy (e.g., focal loss instead of CrossEntropy, mixup augmentation, adjust class weight formula, add early stopping patience)"
    ],
    "summary": "2-3 sentence executive summary: current model state, trend vs previous run, single highest-priority improvement action."
    }}"""

    system = (
        "You are a senior ML engineer. You provide specific, implementable recommendations "
        "grounded in the actual model architecture and data pipeline described. "
        "You respond ONLY with valid JSON."
    )

    raw = await ollama_generate(prompt, system)
    result = _safe_parse_json(
        raw,
        fallback={
            "data_generation": [],
            "model": [],
            "training": [],
            "summary": raw[:300],
        },
    )

    # Ollama ponekad vrati listu dictova umesto listu stringova — normalizacija
    for key in ("data_generation", "model", "training"):
        items = result.get(key, [])
        normalized = []
        for item in items:
            if isinstance(item, dict):
                text = item.get("recommendation") or item.get("text") or item.get("description") or str(item)
                normalized.append(text)
            else:
                normalized.append(str(item))
        result[key] = normalized

    logger.info("[LangGraph] node_synthesize_recommendations: done")

    return {
        "recommendations": {
            "data_generation": result.get("data_generation", []),
            "model":           result.get("model", []),
            "training":        result.get("training", []),
        },
        "summary": result.get("summary", ""),
    }


def node_decision(state: MetricsState) -> dict:
    """
    Deterministički odlučuje o sledećem koraku na osnovu metrika.
    Ne poziva Ollamu — čista Python logika sa fiksnim pragovima.

    Pravila:
        MCC < DECISION_RETRAIN_MCC                     => RETRAIN
        MCC < DECISION_SCAN_MCC ili regression_detected => SCAN_FIRST
        inače                                           => SUGGEST_ONLY
    """
    logger.info("[LangGraph] node_decision: start")

    mcc        = state["mcc_score"]
    regression = state.get("regression_detected", False)

    if mcc < DECISION_RETRAIN_MCC:
        decision = "RETRAIN"
    elif mcc < DECISION_SCAN_MCC or regression:
        decision = "SCAN_FIRST"
    else:
        decision = "SUGGEST_ONLY"

    logger.info(
        f"[LangGraph] node_decision: MCC={mcc:.4f} | "
        f"regression={regression} | decision={decision}"
    )

    return {"decision": decision}


def _route_after_decision(state: MetricsState) -> str:
    """
    Routing funkcija za conditional edge posle node_decision.
    Vraca naziv sledeceg cvora kao string.
    """
    return state["decision"]


def _find_file_recursive(filename: str, root: Path) -> Path | None:
    """
    Rekurzivno traži fajl po imenu unutar root direktorijuma.
    Preskače foldere definisane u SCAN_SKIP_DIRS.
    Vraća prvu pronađenu putanju ili None ako fajl ne postoji.
    """
    for path in root.rglob(filename):
        # Proverava da li je bilo koji deo putanje u skip listi
        if any(part in SCAN_SKIP_DIRS for part in path.parts):
            continue
        return path
    return None


async def node_scan_codebase(state: MetricsState) -> dict:
    """
    Rekurzivno skenira relevantne fajlove iz root direktorijuma projekta.
    Koje fajlove čita određuje se na osnovu slabih klasa iz prethodne analize.

    Uvek čita: hyperparam.py, config.py
    Po slaboj klasi: attacks.py, dataset_generator.py, windowing.py, normal.py

    Preskače foldere: __pycache__, venv, .venv, .git, .idea
    Sadržaj se skraćuje na SCAN_MAX_CHARS_PER_FILE da ne bi prekoračio
    kontekst prozor Ollame.

    Popunjava: scanned_files
    """
    logger.info("[LangGraph] node_scan_codebase: start")

    # Određivanje koje fajlove treba skenirati
    files_to_scan: set[str] = set(ALWAYS_SCAN)
    for cls in state.get("weak_classes", []):
        files_to_scan.update(CLASS_TO_FILES.get(cls, []))

    logger.info(f"  Target files: {sorted(files_to_scan)}")
    logger.info(f"  Skipping dirs: {SCAN_SKIP_DIRS}")

    scanned: dict[str, str] = {}
    root = Path(".")

    for filename in sorted(files_to_scan):
        filepath = _find_file_recursive(filename, root)

        if filepath is None:
            logger.warning(f"  Not found anywhere in project: {filename}")
            scanned[filename] = f"[FILE NOT FOUND: {filename}]"
            continue

        try:
            content = filepath.read_text(encoding="utf-8")
            # Skraćivanje ako je fajl prevelik
            if len(content) > SCAN_MAX_CHARS_PER_FILE:
                content = (
                    content[:SCAN_MAX_CHARS_PER_FILE]
                    + f"\n... [truncated — {len(content)} total chars]"
                )
            # Ključ uključuje relativnu putanju radi jasnoće (npr. "api/config.py")
            rel_key = str(filepath.relative_to(root))
            scanned[rel_key] = content
            logger.info(f"  Read {rel_key}: {len(content)} chars")
        except Exception as e:
            logger.warning(f"  Could not read {filepath}: {e}")
            scanned[filename] = f"[READ ERROR: {e}]"

    logger.info(f"[LangGraph] node_scan_codebase: done | {len(scanned)} files scanned")

    return {"scanned_files": scanned}


# Definicija prostora pretrage — mora odgovarati hyperparam.py
# Koristi se za validaciju LLM odgovora i formatiranje prompta
_HYPERPARAMETER_SPACE = {
    "hidden_size":   {"type": "choice",   "values": [64, 128, 256]},
    "num_layers":    {"type": "choice",   "values": [1, 2]},
    "dropout":       {"type": "range",    "min": 0.1, "max": 0.4},
    "learning_rate": {"type": "range",    "min": 1e-4, "max": 1e-2},
    "seq_len":       {"type": "choice",   "values": [20, 30, 50]},
}


def _validate_hyperparams(proposed: dict) -> dict:
    """
    Proverava da li su predložene vrednosti u okviru dozvoljenog prostora.
    Vrednosti van opsega se zamenjuju najbližom validnom vrednošću.
    Vraća ispravljeni rečnik.
    """
    validated = {}
    for param, spec in _HYPERPARAMETER_SPACE.items():
        raw = proposed.get(param)
        if raw is None:
            logger.warning(f"  Missing proposed value for '{param}' — skipping.")
            continue

        if spec["type"] == "choice":
            if raw not in spec["values"]:
                # Uzima najbližu vrednost iz liste
                closest = min(spec["values"], key=lambda v: abs(v - raw))
                logger.warning(
                    f"  '{param}' value {raw} not in {spec['values']} "
                    f"— clamped to {closest}."
                )
                validated[param] = closest
            else:
                validated[param] = raw

        elif spec["type"] == "range":
            clamped = max(spec["min"], min(spec["max"], float(raw)))
            if clamped != raw:
                logger.warning(
                    f"  '{param}' value {raw} out of range "
                    f"[{spec['min']}, {spec['max']}] — clamped to {clamped}."
                )
            validated[param] = round(clamped, 6)

    return validated


async def node_propose_hyperparams(state: MetricsState) -> dict:
    """
    Poziva Ollamu da predloži konkretne vrednosti hyperparametara na osnovu:
    - rezultata analize metrika (slabe klase, preporuke)
    - sadržaja skeniranih fajlova (ako postoje)
    - striktno definisanog prostora pretrage (_HYPERPARAMETER_SPACE)

    LLM je ograničen da bira isključivo iz dozvoljenih vrednosti.
    Odgovor se validira kroz _validate_hyperparams() pre upisivanja u state.

    Popunjava: proposed_hyperparams
    """
    logger.info("[LangGraph] node_propose_hyperparams: start")

    # Formatiranje prostora pretrage za prompt
    space_lines = []
    for param, spec in _HYPERPARAMETER_SPACE.items():
        if spec["type"] == "choice":
            space_lines.append(f"  - {param}: choose ONE from {spec['values']}")
        else:
            space_lines.append(
                f"  - {param}: float in range [{spec['min']}, {spec['max']}]"
            )
    space_str = "\n".join(space_lines)

    # Kontekst skeniranih fajlova (samo nazivi ako je sadrzaj prevelik)
    scanned = state.get("scanned_files", {})
    if scanned:
        scan_ctx = "Scanned project files (relevant excerpts):\n"
        for fname, content in scanned.items():
            if not content.startswith("["):
                scan_ctx += f"\n--- {fname} ---\n{content[:800]}\n"
    else:
        scan_ctx = "No scanned files available."

    prompt = f"""You are tuning hyperparameters for a DDoS detection LSTM model.

        ANALYSIS SUMMARY:
        - MCC score: {state['mcc_score']:.4f}
        - Decision: {state.get('decision', 'N/A')}
        - Weak classes: {', '.join(state.get('weak_classes', [])) or 'none'}
        - Key findings: {state.get('per_class_analysis', '')[:400]}
        - Recommendations: {str(state.get('recommendations', {}))[:400]}

        {scan_ctx}

        HYPERPARAMETER SPACE (you MUST stay within these bounds):
        {space_str}

        Based on the analysis, propose the best hyperparameter values to improve model performance.
        For "choice" parameters, pick ONLY one of the listed values.
        For "range" parameters, pick a float within the stated range.

        Respond ONLY with a valid JSON object, example:
        {{
        "hidden_size": 256,
        "num_layers": 2,
        "dropout": 0.25,
        "learning_rate": 0.001,
        "seq_len": 30,
        "reasoning": "Short explanation (2-3 sentences) why these values address the identified weak classes."
        }}"""

    system = (
        "You are an ML engineer specializing in LSTM hyperparameter optimization. "
        "You always respect the given hyperparameter space boundaries. "
        "You respond ONLY with valid JSON."
    )

    raw = await ollama_generate(prompt, system)
    result = _safe_parse_json(
        raw,
        fallback={
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.3,
            "learning_rate": 0.001,
            "seq_len": 30,
            "reasoning": "Fallback defaults — LLM response could not be parsed.",
        },
    )

    reasoning = result.pop("reasoning", "")
    validated  = _validate_hyperparams(result)

    logger.info(
        f"[LangGraph] node_propose_hyperparams: done | "
        f"proposed={validated} | reasoning='{reasoning[:100]}'"
    )

    # Čuvamo reasoning kao posebno polje radi prikaza korisniku
    validated["_reasoning"] = reasoning

    return {"proposed_hyperparams": validated}

async def node_human_confirm(state: MetricsState) -> dict:
    """
    Blokira izvršavanje grafa i čeka potvrdu korisnika u terminalu.
    Koristi run_in_executor da input() ne blokira async event loop.
    Popunjava: human_confirmed
    """
    proposed = state.get("proposed_hyperparams", {})
    reasoning = proposed.get("_reasoning", "")

    print("\n" + "=" * 65)
    print("  RETRAIN CONFIRMATION REQUIRED")
    print("=" * 65)
    print(f"\n  Decision: {state.get('decision')} | MCC: {state['mcc_score']:.4f}")
    if reasoning:
        print(f"\n  Reasoning: {reasoning}")
    print("\n  Proposed hyperparameters:")
    for param, value in proposed.items():
        if param == "_reasoning":
            continue
        print(f"    {param:<20} {value}")
    print()

    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(
        None,
        lambda: input("  Proceed with retraining? [y/N]: ").strip().lower()
    )
    confirmed = answer in ("y", "yes")
    logger.info(f"[LangGraph] node_human_confirm: confirmed={confirmed}")
    return {"human_confirmed": confirmed}


def _route_after_confirm(state: MetricsState) -> str:
    """Routing posle human_confirm — retrain ili skip."""
    return "retrain" if state["human_confirmed"] else "skip"


def _patch_hyperparam_file(hp_path: Path, proposed: dict) -> None:
    """
    Upisuje predložene hyperparametre u hyperparam.py regex zamenom.
    Pravi .bak backup pre izmene.
    """
    import re
    content = hp_path.read_text(encoding="utf-8")
    backup = hp_path.with_suffix(".py.bak")
    shutil.copy(hp_path, backup)
    logger.info(f"  Backed up hyperparam.py -> {backup}")

    for param, value in proposed.items():
        spec = _HYPERPARAMETER_SPACE.get(param)
        if spec is None:
            continue
        if spec["type"] == "choice":
            pattern     = rf'("{param}"\s*:\s*)\[[^\]]*\]'
            replacement = rf'\g<1>[{value}]'
        else:
            pattern     = rf'("{param}"\s*:\s*)\([^)]*\)'
            replacement = rf'\g<1>({value}, {value})'

        new_content, n = re.subn(pattern, replacement, content)
        if n:
            logger.info(f"  Patched '{param}' → {value}")
            content = new_content
        else:
            logger.warning(f"  Could not patch '{param}' — pattern not matched.")
            
        # Poseban slučaj za string vrednosti kao što je MODEL_NAME
        if isinstance(value, str):
            pattern     = rf'({re.escape(param)}\s*=\s*)["\'][^"\']*["\']'
            replacement = rf'\g<1>"{value}"'
            new_content, n = re.subn(pattern, replacement, content)
            if n:
                logger.info(f"  Patched '{param}' → {value}")
                content = new_content
            else:
                logger.warning(f"  Could not patch '{param}' — pattern not matched.")
            continue

    hp_path.write_text(content, encoding="utf-8")


async def node_trigger_retrain(state: MetricsState) -> dict:
    """
    Pronalazi torch_nn.py, patchuje hyperparam.py (uključujući novi MODEL_NAME)
    i pokreće retraining. Čeka završetak i streamuje output liniju po liniju.
    Popunjava: retrain_triggered, retrain_command
    """
    import datetime

    proposed = {k: v for k, v in state["proposed_hyperparams"].items()
                if k != "_reasoning"}

    # Generisanje jedinstvenog imena modela sa timestampom
    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"ddos_lstm_retrain_{timestamp}.pt"
    proposed["MODEL_NAME"] = model_name
    logger.info(f"  New model will be saved as: {model_name}")

    hp_path = _find_file_recursive("hyperparam.py", Path("."))
    if hp_path:
        _patch_hyperparam_file(hp_path, proposed)
    else:
        logger.warning("  hyperparam.py not found — using existing values.")

    torch_path = _find_file_recursive("torch_nn.py", Path("."))
    if torch_path is None:
        logger.error("  torch_nn.py not found — cannot trigger retrain.")
        return {"retrain_triggered": False, "retrain_command": "NOT FOUND"}

    # -u flag forsira unbuffered stdout — neophodan za real-time streaming
    command     = ["python", "-u", str(torch_path)]
    command_str = " ".join(command)
    print(f"\n  Starting retrain: {command_str}")
    print(f"  Output model: {model_name}\n")

    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        print(f"  PID: {proc.pid}\n")

        async for line in proc.stdout:
            print(f"  [train] {line.decode().rstrip()}", flush=True)

        await proc.wait()
        success = proc.returncode == 0

        if success:
            print(f"\n  Retrain finished successfully (exit code 0).")
            print(f"  Model saved as: {model_name}")
        else:
            print(f"\n  Retrain finished with errors (exit code {proc.returncode}).")

    except Exception as e:
        logger.error(f"  Failed to start subprocess: {e}")
        return {"retrain_triggered": False, "retrain_command": command_str}

    return {
        "retrain_triggered": success,
        "retrain_command":   command_str,
    }

def build_analyzer_graph():
    """
    Kreira i kompajlira LangGraph graf za analizu metrika.

    Trenutni tok (Korak 3):
        load_and_compare => analyze_per_class => analyze_confusion => synthesize
            => decision
                ├── SUGGEST_ONLY => END
                ├── SCAN_FIRST   => scan_codebase => propose_hyperparams => END (Korak 4 dodaje human_confirm)
                └── RETRAIN      => propose_hyperparams => END (Korak 4 dodaje human_confirm)
    """
    graph = StateGraph(MetricsState)

    # --- Postojeći čvorovi ---
    graph.add_node("load_and_compare",    node_load_and_compare_metrics)
    graph.add_node("analyze_per_class",   node_analyze_per_class)
    graph.add_node("analyze_confusion",   node_analyze_confusion)
    graph.add_node("synthesize",          node_synthesize_recommendations)

    # --- Čvorovi Koraka 2 ---
    graph.add_node("decision",            node_decision)
    graph.add_node("scan_codebase",       node_scan_codebase)

    # --- Čvorovi Koraka 3 ---
    graph.add_node("propose_hyperparams", node_propose_hyperparams)

    # --- Čvorovi Koraka 4 ---
    graph.add_node("human_confirm",       node_human_confirm)
    graph.add_node("trigger_retrain",     node_trigger_retrain)

    # --- Sekvencijalne grane ---
    graph.set_entry_point("load_and_compare")
    graph.add_edge("load_and_compare",    "analyze_per_class")
    graph.add_edge("analyze_per_class",   "analyze_confusion")
    graph.add_edge("analyze_confusion",   "synthesize")
    graph.add_edge("synthesize",          "decision")

    # --- Conditional edges posle decision ---
    graph.add_conditional_edges(
        "decision",
        _route_after_decision,
        {
            "SUGGEST_ONLY": END,
            "SCAN_FIRST":   "scan_codebase",
            "RETRAIN":      "propose_hyperparams",   # preskace scan, ide direktno
        },
    )

    # SCAN_FIRST grana: scan => propose
    graph.add_edge("scan_codebase",       "propose_hyperparams")

    # propose_hyperparams trenutno završava na END (Korak 4 dodaje human_confirm)
    graph.add_edge("propose_hyperparams", "human_confirm")

    graph.add_conditional_edges(
        "human_confirm",
        _route_after_confirm,
        {"retrain": "trigger_retrain", "skip": END},
    )

    graph.add_edge("trigger_retrain", END)

    return graph.compile()


# Globalna instanca deli se kroz celu aplikaciju
analyzer_graph = build_analyzer_graph()