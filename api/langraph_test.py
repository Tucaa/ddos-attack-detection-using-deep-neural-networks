import asyncio
import json
import os
import sys
import time
from pathlib import Path


# Funkcija za interaktivni unos od strane korisnika
def get_user_input():
    print("=" * 65)
    print("  LANGGRAPH ANALYZER — KONFIGURACIJA")
    print("=" * 65)
    
    # Unos putanje za metrike
    metrics_input = input("Unesite putanju do fajla sa metrikama (default: results/test_metrics.json): ").strip()
    metrics_path = metrics_input if metrics_input else "results/test_metrics.json"
    
    # Unos Ollama URL-a
    ollama_input = input("Unesite Ollama host URL (default: OLLAMA_HOST env var ili http://localhost:11434): ").strip()
    ollama_url = ollama_input if ollama_input else None
    
    return metrics_path, ollama_url


# Pomocne funkcije za ispis
def _separator(char="=", width=65):
    print(char * width)

def _section(title: str):
    print()
    _separator()
    print(f"  {title}")
    _separator()

def _print_list(items: list[str], indent: int = 4):
    for i, item in enumerate(items, 1):
        prefix = " " * indent + f"{i}."
        words = item.split()
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

def _print_results(state: dict):
    """Stampa rezultate LangGraph analize u citljivom formatu."""
    _section("LANGGRAPH ANALYSIS RESULTS")

    # Slabe klase
    print("\n  Weak classes (f1 < 0.80 | recall < 0.75 | roc_auc < 0.85):")
    if state["weak_classes"]:
        for cls in state["weak_classes"]:
            print(f"    ! {cls}")
    else:
        print("    No weak classes identified.")

    # Per-class analiza
    _section("PER-CLASS ANALYSIS")
    print(state.get("per_class_analysis", "(empty)"))

    # Confusion matrix analiza
    _section("CONFUSION MATRIX ANALYSIS")
    print(state.get("confusion_analysis", "(empty)"))

    # Preporuke
    recs = state.get("recommendations", {})

    _section("RECOMMENDATIONS — Data generation")
    _print_list(recs.get("data_generation", []))

    _section("RECOMMENDATIONS — Model architecture/hyperparameters")
    _print_list(recs.get("model", []))

    _section("RECOMMENDATIONS — Training strategy")
    _print_list(recs.get("training", []))

    _section("CONCLUSION")
    print(state.get("summary", "(empty)"))

    print()
    _separator()


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
    _separator()
    print("  DDoS LSTM — LangGraph Analyzer Test")
    _separator()

    # Provera Ollama
    print("\n[1/4] Checking Ollama...")
    ollama_ok = await check_ollama(ollama_host)
    if not ollama_ok:
        print(f"\n  Hint: If Ollama is in Docker, try running again and type:")
        print(f"    http://ollama:11434 ili http://localhost:11434")
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
    print("  Nodes: analyze_per_class → analyze_confusion → synthesize")
    print("  (Each node calls Ollama — this may take 30-90s)\n")

    initial_state = {
        "classification_report": metrics["classification_report"],
        "confusion_matrix":      metrics["confusion_matrix"],
        "mcc_score":              metrics["mcc_score"],
        "roc_auc_scores":        metrics["roc_auc_scores"],
        "class_labels":          metrics["class_labels"],
        # Medjurezultati
        "per_class_analysis":    "",
        "confusion_analysis":    "",
        # Izlaz
        "weak_classes":          [],
        "recommendations":       {},
        "summary":               "",
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

    # Rezultati
    print(f"[4/4] Graph finished in {elapsed:.1f}s")
    _print_results(final_state)

    # Cuvanje rezultata
    output_path = Path(metrics_path).parent / "langgraph_analysis.json"
    output = {
        "weak_classes":       final_state["weak_classes"],
        "per_class_analysis": final_state["per_class_analysis"],
        "confusion_analysis": final_state["confusion_analysis"],
        "recommendations":    final_state["recommendations"],
        "summary":            final_state["summary"],
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