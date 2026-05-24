import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path


# Funkcija za interaktivni unos od strane korisnika
def get_user_input():
    print("=" * 65)
    print("  LANGGRAPH ANALYZER — CONFIGURATION")
    print("=" * 65)
    
    # Unos putanje za metrike
    metrics_input = input("Enter path to metrics file (default: results/test_metrics.json): ").strip()
    metrics_path = metrics_input if metrics_input else "results/test_metrics.json"
    
    # Unos Ollama URL-a
    ollama_input = input("Enter Ollama host url (default: OLLAMA_HOST env var ili http://localhost:11434): ").strip()
    ollama_url = ollama_input if ollama_input else None
    
    return metrics_path, ollama_url


# Pomocne funkcije za ispis
def separator(char="=", width=65):
    print(char * width)

def section(title: str):
    print()
    separator()
    print(f"  {title}")
    separator()

def print_list(items: list, indent: int = 4):
    # Konvertuje stavke u string ako LLM vrati dict umesto plain teksta
    for i, item in enumerate(items, 1):
        if isinstance(item, dict):
            text = item.get("recommendation") or item.get("text") or item.get("description") or str(item)
        else:
            text = str(item)
        prefix = " " * indent + f"{i}."
        words = text.split()
        line, lines = [], []
        for word in words:
            if len(" ".join(line + [word])) > 90:
                lines.append(" ".join(line))
                line = [word]
            else:
                line.append(word)
        if line:
            lines.append(" ".join(line))
        print(f"{prefix} {lines[0]}")
        for continuation in lines[1:]:
            print(" " * (indent + 4) + continuation)

def print_results(state: dict):
    """Stampa rezultate LangGraph analize u citljivom formatu."""
    section("LANGGRAPH ANALYSIS RESULTS")

    # Delta metrike vs prethodni run
    delta = state.get("metrics_delta", {})
    if delta.get("mcc_delta") is not None:
        print("\n  Comparison with previous run:")
        mcc_d = delta["mcc_delta"]
        f1_d  = delta.get("macro_f1_delta")
        reg   = state.get("regression_detected", False)

        mcc_arrow = "^" if mcc_d >= 0 else "v"
        print(f"    MCC:      {mcc_arrow} {mcc_d:+.4f}")
        if f1_d is not None:
            f1_arrow = "^" if f1_d >= 0 else "v"
            print(f"    Macro F1: {f1_arrow} {f1_d:+.4f}")

        per_cls = delta.get("per_class_f1_delta", {})
        if per_cls:
            regressions = [(c, v) for c, v in per_cls.items() if v < -0.03]
            if regressions:
                print("    Regressions (F1 drop > 0.03):")
                for cls, v in regressions:
                    print(f"      ! {cls}: {v:+.3f}")

        if reg:
            print("    *** REGRESSION DETECTED — model is worse than previous run ***")
        else:
            print("    No significant regression detected.")
    else:
        print("\n  Comparison with previous run: N/A (first run or no baseline)")

    # Slabe klase
    print("\n  Weak classes (f1 < 0.80 | recall < 0.75 | roc_auc < 0.85):")
    if state["weak_classes"]:
        for cls in state["weak_classes"]:
            print(f"    ! {cls}")
    else:
        print("    No weak classes identified.")

    # Per-class analiza
    section("PER-CLASS ANALYSIS")
    print(state.get("per_class_analysis", "(empty)"))

    # Confusion matrix analiza
    section("CONFUSION MATRIX ANALYSIS")
    print(state.get("confusion_analysis", "(empty)"))

    # Preporuke
    recs = state.get("recommendations", {})

    section("RECOMMENDATIONS — Data generation")
    print_list(recs.get("data_generation", []))

    section("RECOMMENDATIONS — Model architecture/hyperparameters")
    print_list(recs.get("model", []))

    section("RECOMMENDATIONS — Training strategy")
    print_list(recs.get("training", []))

    # Decision rezultat — pre zaključka jer postavlja kontekst
    decision = state.get("decision", "")
    if decision:
        section("DECISION")
        labels = {
            "SUGGEST_ONLY": "Suggestions only — model performance is acceptable.",
            "SCAN_FIRST":   "Codebase scanned — review proposed changes before retraining.",
            "RETRAIN":      "Retraining recommended — MCC below critical threshold.",
        }
        print(f"  => {decision}: {labels.get(decision, '')}")

    # Predloženi hyperparametri
    proposed = state.get("proposed_hyperparams", {})
    if proposed:
        section("PROPOSED HYPERPARAMETERS")
        reasoning = proposed.get("_reasoning", "")
        if reasoning:
            print(f"  Reasoning: {reasoning}\n")
        for param, value in proposed.items():
            if param == "_reasoning":
                continue
            print(f"    {param:<20} {value}")

    scanned = state.get("scanned_files", {})  
    if scanned:
        print(f"\n  Scanned files ({len(scanned)}):")
        for fname, content in scanned.items():
            if content.startswith("["):
                print(f"[ERROR] {fname}")
            else:
                print(f"[OK] {fname}  ({len(content)} chars)")

    section("CONCLUSION")
    print(state.get("summary", "(empty)"))

    # Retrain status
    if state.get("retrain_triggered"):
        print(f"\n  Retrain started | command: {state.get('retrain_command', '')}")
    elif state.get("human_confirmed") is False and state.get("proposed_hyperparams"):
        print("\n  Retrain skipped — user declined.")

    print()
    separator()


# Provera Ollama dostupnosti
async def check_ollama(host: str):
    """Proverava da li je Ollama dostupan pre pokretanja grafa."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{host}/api/tags")
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                print(f"  Ollama available at: {host}")
                print(f"  Available models: {', '.join(models) if models else '(none)'}")
                return True
            else:
                print(f"  Ollama responded with status {resp.status_code}")
                return False
    except Exception as e:
        print(f"  Ollama is NOT available at {host}: {e}")
        return False


# Glavni test
async def run_test(metrics_path: str, ollama_url: str):
    # Postavljamo Ollama host pre importa ollama_client-a ako je korisnik uneo vrednost
    if ollama_url:
        os.environ["OLLAMA_HOST"] = ollama_url
        print(f"Ollama host override: {ollama_url}")

    # Import grafa
    try:
        from langraph import analyzer_graph
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run the script from the root directory of the project (where the api/ folder is located).")
        sys.exit(1)

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    print()
    separator()
    print("  DDoS LSTM — LangGraph Analyzer Test")
    separator()

    # Provera Ollama
    print("\n[1/4] Checking Ollama...")
    ollama_ok = await check_ollama(ollama_host)
    if not ollama_ok:
        print(f"\n  Hint: If Ollama is in Docker, try running again and type:")
        print(f"    http://ollama:11434 or http://localhost:11434")
        sys.exit(1)

    # Ucitavanje metrika
    print(f"\n[2/4] Loading metrics from: {metrics_path}")
    p = Path(metrics_path)
    if not p.exists():
        print(f"  File not found: {metrics_path}")
        print("  Generate it by running the training script, or use test_metrics.json")
        sys.exit(1)

    with open(p, "r") as f:
        metrics = json.load(f)

    print(f"  MCC score:     {metrics['mcc_score']:.4f}")
    print(f"  Classes:          {len(metrics['class_labels'])}")
    print(f"  Timestamp:     {metrics.get('timestamp', 'N/A')}")
    if "summary" in metrics:
        s = metrics["summary"]
        print(f"  Macro F1:       {s.get('macro_f1', 0):.4f}")
        print(f"  Total samples: {s.get('total_samples', 0):,}")

    # Pravljenje inicijalnog stanja
    print("\n[3/4] Running LangGraph graph...")
    print("  Nodes: load_and_compare => analyze_per_class => analyze_confusion => synthesize => decision => [scan_codebase] => propose_hyperparams => human_confirm => [trigger_retrain]")
    print("  (Ollama nodes may take 30-90s)\n")

    initial_state = {
        # Ulazne metrike (trenutni run)
        "classification_report": metrics["classification_report"],
        "confusion_matrix":      metrics["confusion_matrix"],
        "mcc_score":             metrics["mcc_score"],
        "roc_auc_scores":        metrics["roc_auc_scores"],
        "class_labels":          metrics["class_labels"],

        # Poređenje — puni ih node_load_and_compare_metrics
        "previous_metrics":    {},
        "metrics_delta":       {},
        "regression_detected": False,

        # Medjurezultati analize
        "per_class_analysis":  "",
        "confusion_analysis":  "",

        # Izlaz analize
        "weak_classes":    [],
        "recommendations": {},
        "summary":         "",

        # Decision i akcija — popunjavaju se u kasnijim koracima
        "decision":             "",
        "scanned_files":        {},
        "proposed_hyperparams": {},
        "human_confirmed":      False,
        "retrain_triggered":    False,
        "retrain_command":      "",
    }

    t_start = time.time()
    try:
        final_state = await analyzer_graph.ainvoke(initial_state)
    except Exception as e:
        print(f"  ERROR in graph: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t_start

    # Backup trenutnih metrika u _prev — radi se POSLE uspesnog run-a
    # kako bi sledeci run imao tacne prethodne metrike za poredenje
    prev_path = p.parent / "test_metrics_prev.json"
    shutil.copy(p, prev_path)
    print(f"  Metrics backed up for next run -> {prev_path}")

    # Rezultati
    print(f"[4/4] Graph finished in {elapsed:.1f}s")
    print_results(final_state)

    # Cuvanje rezultata
    output_path = Path(metrics_path).parent / "langgraph_analysis.json"
    output = {
        # Analiza
        "weak_classes":       final_state["weak_classes"],
        "per_class_analysis": final_state["per_class_analysis"],
        "confusion_analysis": final_state["confusion_analysis"],
        "recommendations":    final_state["recommendations"],
        "summary":            final_state["summary"],
        # Decision i skeniranje
        "decision":             final_state.get("decision", ""),
        "scanned_files":        list(final_state.get("scanned_files", {}).keys()),
        "proposed_hyperparams": final_state.get("proposed_hyperparams", {}),
        # Delta vs prethodni run
        "metrics_delta":      final_state.get("metrics_delta", {}),
        "regression_detected": final_state.get("regression_detected", False),
        # Meta
        "elapsed_seconds":    round(elapsed, 2),
        "source_metrics":     str(metrics_path),
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Analysis saved -> {output_path}")


if __name__ == "__main__":
    # Prvo pitamo korisnika za input, pa prosledjujemo u glavnu funkciju
    metrics_file, ollama_server = get_user_input()
    asyncio.run(run_test(metrics_file, ollama_server))