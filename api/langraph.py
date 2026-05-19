import json
import logging
import shutil
from pathlib import Path
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from ollama_client import ollama_generate, _extract_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

METRICS_PATH      = Path("results/test_metrics.json")
PREV_METRICS_PATH = Path("results/test_metrics_prev.json")


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
    proposed_hyperparams: dict    # predlozene vrednosti iz HYPERPARAMETER_SPACE
    human_confirmed: bool         # da li je korisnik potvrdio retrain
    retrain_triggered: bool       # da li je retrain pokrenut
    retrain_command: str          # komanda koja je izvrsena (za logovanje)



def _format_per_class_metrics(report: dict, roc_auc: dict, mcc: float) -> str:
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
                f"  Previous metrics loaded: MCC={previous.get('mcc_score', 'N/A')}, "
                f"timestamp={previous.get('timestamp', 'N/A')}"
            )
        except Exception as e:
            logger.warning(f"  Could not load previous metrics: {e} — treating as first run.")
            previous = {}
    else:
        logger.info("  No previous metrics file found — this is treated as first baseline run.")

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
            logger.warning(f"  Per-class regressions: {', '.join(regressions)}")

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

    logger.info("[LangGraph] node_synthesize_recommendations: done")

    return {
        "recommendations": {
            "data_generation": result.get("data_generation", []),
            "model":           result.get("model", []),
            "training":        result.get("training", []),
        },
        "summary": result.get("summary", ""),
    }


def build_analyzer_graph():
    """
    Kreira i kompajlira LangGraph graf za analizu metrika.

    Trenutni tok (Korak 1):
        load_and_compare → analyze_per_class → analyze_confusion → synthesize → END

    Buduci tok (Koraci 2-4):
        load_and_compare → analyze_per_class → analyze_confusion → synthesize
            → decision (conditional edges)
                ├── SUGGEST_ONLY → END
                ├── SCAN_FIRST   → scan_codebase → propose_hyperparams → human_confirm → [END | retrain]
                └── RETRAIN      → propose_hyperparams → human_confirm → [END | retrain]
    """
    graph = StateGraph(MetricsState)

    graph.add_node("load_and_compare",  node_load_and_compare_metrics)
    graph.add_node("analyze_per_class", node_analyze_per_class)
    graph.add_node("analyze_confusion", node_analyze_confusion)
    graph.add_node("synthesize",        node_synthesize_recommendations)

    graph.set_entry_point("load_and_compare")
    graph.add_edge("load_and_compare",  "analyze_per_class")
    graph.add_edge("analyze_per_class", "analyze_confusion")
    graph.add_edge("analyze_confusion", "synthesize")
    graph.add_edge("synthesize",        END)

    return graph.compile()


# Globalna instanca deli se kroz celu aplikaciju
analyzer_graph = build_analyzer_graph()