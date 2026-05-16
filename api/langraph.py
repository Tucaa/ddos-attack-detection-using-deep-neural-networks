import json
import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END

from ollama_client import ollama_generate, _extract_json

logger = logging.getLogger(__name__)



# Stanje grafa
class MetricsState(TypedDict):
    # Ulaz 
    classification_report: dict   # {klasa: {precision, recall, f1-score, support}}
    confusion_matrix: list        # 2D lista integera (num_classes x num_classes)
    mcc_score: float
    roc_auc_scores: dict          # {klasa: float}
    class_labels: list            # lista naziva klasa u ispravnom redosledu

    # Medjurezultati 
    per_class_analysis: str
    confusion_analysis: str

    # Izlaz 
    weak_classes: list            # nazivi klasa sa losim performansama
    recommendations: dict         # {data_generation: [], model: [], training: []}
    summary: str



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

    prompt = f"""You are analyzing per-class performance of a DDoS traffic classification LSTM model.

    The model classifies network traffic windows into 9 classes:
    - normal traffic
    - 8 DDoS attack types: udp_flood_large, dns_amplification, subnet_carpet_bombing,
    syn_flood, icmp_flood, udp_flood_mixed, ntp_amplification, ack_flood

    Per-class metrics:
    {metrics_str}

    Thresholds for "weak": f1-score < 0.80 OR recall < 0.75 OR roc_auc < 0.85

    Respond ONLY with a valid JSON object (no markdown, no explanation outside JSON):
    {{
    "weak_classes": ["list of class names that fall below thresholds"],
    "analysis": "Technical paragraph (4-6 sentences) explaining WHY these specific classes underperform. Consider: similar traffic patterns between attack types, feature overlap (e.g., udp_flood_mixed vs udp_flood_large share packet size distributions), class imbalance, insufficient diversity in synthetic data generation, or transition period blending artifacts."
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

    prompt = f"""You are synthesizing actionable improvement recommendations for a DDoS LSTM classifier.

    CONTEXT:
    - Model: 2-layer LSTM with attention, sliding window of 20 samples x 12 features
    - Features: packet_size, packets_per_second, bytes_per_second, src_port, dst_port,
    protocol, tcp_flags, ttl, inter_arrival_time, flow_duration, unique_src_ips, unique_dst_ports
    - Data: synthetic traffic with Gaussian noise, transition period blending, minority class oversampling
    - Training: CrossEntropyLoss with inverse-frequency class weights, Adam optimizer, ReduceLROnPlateau

    ANALYSIS RESULTS:
    - Overall MCC: {state["mcc_score"]:.4f}
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
    "summary": "2-3 sentence executive summary: current model state, biggest bottleneck, single highest-priority improvement action."
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
    Sekvencijalan tok: per_class -> confusion -> synthesize
    """
    graph = StateGraph(MetricsState)

    graph.add_node("analyze_per_class", node_analyze_per_class)
    graph.add_node("analyze_confusion",  node_analyze_confusion)
    graph.add_node("synthesize",         node_synthesize_recommendations)

    graph.set_entry_point("analyze_per_class")
    graph.add_edge("analyze_per_class", "analyze_confusion")
    graph.add_edge("analyze_confusion",  "synthesize")
    graph.add_edge("synthesize",         END)

    return graph.compile()


# Globalna instanca deli se kroz celu aplikaciju
analyzer_graph = build_analyzer_graph()