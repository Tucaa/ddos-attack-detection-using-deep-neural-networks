import argparse
import asyncio
import json
import sys
import time
import re
from config import CLASS_LABELS, FEATURE_NAMES

import httpx

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_API_HOST    = "http://localhost:8000"
DEFAULT_ATTACK_TYPE = "syn_flood"
TIMEOUT             = 300.0
WINDOW_SIZE         = 20


# --- Pomocne funkcije za ispis ---

def ok(msg):   print(f"  [OK]   {msg}")
def fail(msg): print(f"  [FAIL] {msg}")
def info(msg): print(f"  [INFO] {msg}")

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def show_block(label: str, text: str):
    """Ispisuje sadrzaj bloka sa labelom, bez skracivanja."""
    print(f"\n  --- {label} ---")
    for line in text.strip().splitlines():
        print(f"  {line}")
    print()

def show_token_stats(data: dict):
    """Ispisuje statistiku tokena iz Ollama odgovora."""
    eval_count    = data.get("eval_count", 0)
    eval_duration = data.get("eval_duration", 0)
    prompt_eval   = data.get("prompt_eval_count", 0)
    tps = eval_count / (eval_duration / 1e9) if eval_duration else 0
    info(f"Prompt tokens    : {prompt_eval}")
    info(f"Generated tokens : {eval_count}")
    info(f"Tokens/sec       : {tps:.1f}")


# -----------------------------------------------------------------------
# Testovi
# -----------------------------------------------------------------------

async def test_ping(ollama_host: str) -> bool:
    """Proverava da li je Ollama server dostupan."""
    section("TEST 1 — Ollama ping")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{ollama_host}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", [])
    except httpx.ConnectError:
        fail(f"Cannot connect to: {ollama_host}")
        info("Start Ollama server: 'ollama serve'")
        return False
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")
        return False

    ok(f"Server reachable: {ollama_host}")
    if models:
        ok(f"Available models ({len(models)}):")
        for m in models:
            size_gb = m.get("size", 0) / (1024 ** 3)
            print(f"       - {m['name']}  ({size_gb:.1f} GB)")
    else:
        info("No models loaded.")

    show_block("/api/tags response", json.dumps(data, indent=2))
    return True


async def test_raw_generate(ollama_host: str, model: str) -> bool:
    """Testira osnovni /api/generate poziv."""
    section("TEST 2 — Raw generate")
    info(f"Model: {model}")

    payload = {
        "model": model,
        "prompt": "Reply with exactly one word: hello",
        "system": "Reply with exactly one word.",
        "stream": False,
    }

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(f"{ollama_host}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            response_text = data.get("response", "").strip()
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")
        return False

    elapsed = time.perf_counter() - start
    ok(f"Response in {elapsed:.2f}s")
    show_token_stats(data)
    show_block("Model output", response_text)

    if not response_text:
        fail("Empty response!")
        return False

    ok(f"Response text: '{response_text}'")
    return True


async def test_generate_scenario(ollama_host: str, model: str, attack_type: str) -> bool:
    """
    Testira generisanje matrice saobracaja za zadati tip napada.
    Proverava dimenzije i tip vrednosti.
    """
    section(f"TEST 3 — Scenario matrix ({attack_type})")
    info(f"Requesting {WINDOW_SIZE}x{len(FEATURE_NAMES)} matrix for: {attack_type}")

    features_str = "\n".join(f"  - {name}" for name in FEATURE_NAMES)
    prompt = f"""Generate a synthetic network traffic matrix simulating a **{attack_type}** DDoS attack.
    - Return ONLY a valid JSON array, no explanation, no markdown.
    - Exactly {WINDOW_SIZE} rows, each with exactly {len(FEATURE_NAMES)} float values.
    Features:
    {features_str}
    Respond with ONLY the JSON array."""

    start = time.perf_counter()
    raw = ""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            payload = {
                "model": model,
                "prompt": prompt,
                "system": "You are a network security expert. Respond ONLY with valid JSON arrays.",
                "stream": False,
            }
            resp = await client.post(f"{ollama_host}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("response", "")
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")
        if raw:
            show_block("Partial model output", raw)
        return False

    elapsed = time.perf_counter() - start
    ok(f"Response in {elapsed:.2f}s")
    show_token_stats(data)
    show_block("Raw model output", raw)

    # Uklanjanje markdown omotaca ako postoji
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
    clean = match.group(1).strip() if match else raw.strip()

    try:
        matrix = json.loads(clean)
        ok("JSON parsing successful")
    except json.JSONDecodeError as e:
        fail(f"JSON parsing failed: {e}")
        show_block("Cleaned string that failed to parse", clean)
        return False

    # Provera dimenzija matrice
    if not isinstance(matrix, list):
        fail(f"Expected list, got: {type(matrix).__name__}")
        return False

    if len(matrix) != WINDOW_SIZE:
        fail(f"Expected {WINDOW_SIZE} rows, got: {len(matrix)}")
        return False

    ok(f"Row count: {len(matrix)} / {WINDOW_SIZE}")

    bad_rows = [i for i, row in enumerate(matrix)
                if not isinstance(row, list) or len(row) != len(FEATURE_NAMES)]
    if bad_rows:
        fail(f"Invalid rows ({len(bad_rows)}): {bad_rows[:5]}")
        return False

    ok(f"All rows valid ({len(FEATURE_NAMES)} columns)")

    print("  --- Parsed matrix (all rows) ---")
    for i, row in enumerate(matrix):
        print(f"  Row {i:2}: {row}")
    print()

    return True


async def test_generate_analysis(ollama_host: str, model: str) -> bool:
    """
    Testira generisanje analize napada.
    Proverava da JSON sadrzi 'description' i 'mitigation_steps'.
    Prompt eksplicitno zahteva TACNO ta dva kljuca.
    """
    section("TEST 4 — Attack analysis")

    predicted_class = "syn-flood"
    confidence = 0.87
    info(f"Testing analysis for: {predicted_class} ({confidence:.0%} confidence)")

    # Prompt koji eksplicitno zahteva tacno ta dva kljuca bez varijacija
    prompt = (
        f"Analyze a {predicted_class} DDoS attack detected with {confidence:.2%} confidence.\n\n"
        f"You MUST return a JSON object with EXACTLY these two keys, spelled exactly as shown:\n"
        f'  "description"      — a 3-5 sentence technical description of the attack\n'
        f'  "mitigation_steps" — a JSON list of 5-7 actionable mitigation steps\n\n'
        f"Do NOT use any other key names. Do NOT add extra fields.\n"
        f"Example structure:\n"
        f'{{"description": "...", "mitigation_steps": ["step1", "step2", ...]}}'
    )

    start = time.perf_counter()
    raw = ""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            payload = {
                "model": model,
                "prompt": prompt,
                "system": (
                    "You are a senior network security analyst. "
                    "You respond ONLY with a valid JSON object. "
                    "The JSON must contain exactly two keys: "
                    "'description' (string) and 'mitigation_steps' (array of strings). "
                    "No other keys. No markdown. No explanation outside the JSON."
                ),
                "stream": False,
            }
            resp = await client.post(f"{ollama_host}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("response", "")
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")
        return False

    elapsed = time.perf_counter() - start
    ok(f"Response in {elapsed:.2f}s")
    show_token_stats(data)
    show_block("Raw model output", raw)

    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
    clean = match.group(1).strip() if match else raw.strip()

    try:
        result = json.loads(clean)
        ok("JSON parsing successful")
    except json.JSONDecodeError as e:
        fail(f"JSON parsing failed: {e}")
        show_block("Cleaned string that failed to parse", clean)
        return False

    if "description" not in result or "mitigation_steps" not in result:
        fail(f"Missing required keys. Found: {list(result.keys())}")
        return False

    ok("Keys 'description' and 'mitigation_steps' present")

    show_block("description", result.get("description", ""))
    print("  --- mitigation_steps ---")
    for i, step in enumerate(result.get("mitigation_steps", []), 1):
        print(f"  {i}. {step}")
    print()

    return True


async def test_simulate_endpoint(api_host: str, attack_type: str) -> bool:
    """Testira /simulate endpoint na FastAPI serveru."""
    section(f"TEST 5 — /simulate endpoint ({attack_type})")

    # Provera dostupnosti FastAPI servera
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            health = await client.get(f"{api_host}/health")
            health.raise_for_status()
            health_data = health.json()
    except Exception as e:
        fail(f"FastAPI server not available: {type(e).__name__}: {e}")
        return False

    ok(f"FastAPI server reachable: {api_host}")
    show_block("/health response", json.dumps(health_data, indent=2))

    info(f"Sending request for: {attack_type}")

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(f"{api_host}/simulate", json={"attack_type": attack_type})
    except Exception as e:
        fail(f"HTTP call failed: {type(e).__name__}: {e}")
        return False

    elapsed = time.perf_counter() - start

    if resp.status_code != 200:
        fail(f"HTTP {resp.status_code}: {resp.text[:300]}")
        return False

    ok(f"Response in {elapsed:.2f}s (HTTP 200)")
    data = resp.json()
    show_block("/simulate response", json.dumps(data, indent=2))

    if "analysis" not in data:
        fail("Field 'analysis' missing from response")
        return False

    ok("Field 'analysis' present")
    return True


async def test_edge_cases(api_host: str) -> bool:
    """Testira ocekivano ponasanje sa pogresnim ulazima."""
    section("TEST 6 — Edge cases")

    print("\n  6a — Invalid attack_type on /simulate")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{api_host}/simulate",
                json={"attack_type": "invalid-attack-xyz"}
            )
        show_block("Server response", resp.text)
        if resp.status_code == 400:
            ok("HTTP 400 returned as expected")
        else:
            fail(f"Expected HTTP 400, got: {resp.status_code}")
    except Exception:
        info("FastAPI not available — skipping 6a")

    print("\n  6b — Empty body on /simulate")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{api_host}/simulate", json={})
        show_block("Server response", resp.text)
        if resp.status_code in (400, 422):
            ok(f"HTTP {resp.status_code} returned for empty body")
        else:
            fail(f"Expected 400/422, got: {resp.status_code}")
    except Exception:
        info("FastAPI not available — skipping 6b")

    return True


# -----------------------------------------------------------------------
# Interaktivni mod
# -----------------------------------------------------------------------

MENU = {
    "1": "Ping         — Ollama server availability",
    "2": "Raw generate — basic response test",
    "3": "Scenario     — traffic matrix generation",
    "4": "Analysis     — attack analysis and mitigation",
    "5": "Simulate     — end-to-end API test",
    "6": "Edge cases   — negative scenarios",
    "7": "Run all tests",
}


def _prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"  {text}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        raise KeyboardInterrupt
    return val if val else default


def _pick_attack_type(current: str) -> str:
    print("\n  Available attack types:")
    for i, at in enumerate(CLASS_LABELS, 1):
        marker = " <--" if at == current else ""
        print(f"    {i:2}. {at}{marker}")
    raw = _prompt("Enter number or attack name", current)
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(CLASS_LABELS):
            return CLASS_LABELS[idx]
    return raw if raw in CLASS_LABELS else current


def _show_menu(session: dict):
    print("\n" + "="*55)
    print("  Ollama — Interactive Tests")
    print("="*55)
    print(f"  Ollama  : {session['ollama_host']}")
    print(f"  Model   : {session['model']}")
    print(f"  API     : {session['api_host']}")
    print(f"  Attack  : {session['attack_type']}")
    print()
    for key, label in MENU.items():
        print(f"  {key}. {label}")
    print()
    print("  c. Change configuration")
    print("  q. Exit")
    print()


async def _run_test(choice: str, session: dict):
    """Poziva odgovarajuci test na osnovu odabranog broja."""
    h   = session["ollama_host"]
    m   = session["model"]
    a   = session["attack_type"]
    api = session["api_host"]

    if choice == "1":
        await test_ping(h)
    elif choice == "2":
        await test_raw_generate(h, m)
    elif choice == "3":
        session["attack_type"] = _pick_attack_type(a)
        await test_generate_scenario(h, m, session["attack_type"])
    elif choice == "4":
        await test_generate_analysis(h, m)
    elif choice == "5":
        session["attack_type"] = _pick_attack_type(a)
        await test_simulate_endpoint(api, session["attack_type"])
    elif choice == "6":
        await test_edge_cases(api)
    elif choice == "7":
        session["attack_type"] = _pick_attack_type(a)
        a = session["attack_type"]
        results = [
            await test_ping(h),
            await test_raw_generate(h, m),
            await test_generate_scenario(h, m, a),
            await test_generate_analysis(h, m),
            await test_simulate_endpoint(api, a),
            await test_edge_cases(api),
        ]
        passed = sum(results)
        section(f"RESULT: {passed}/{len(results)} tests passed")


async def interactive_mode(ollama_host: str, api_host: str, model: str):
    """Chat-like REPL za interaktivno testiranje."""
    session = {
        "ollama_host": ollama_host,
        "api_host":    api_host,
        "model":       model,
        "attack_type": DEFAULT_ATTACK_TYPE,
    }

    while True:
        _show_menu(session)
        try:
            choice = _prompt("Selection").lower()
        except KeyboardInterrupt:
            print("\n  Goodbye!\n")
            break

        if choice in ("q", "quit", "exit"):
            print("\n  Goodbye!\n")
            break

        if choice == "c":
            print()
            session["ollama_host"] = _prompt("Ollama host", session["ollama_host"])
            session["model"]       = _prompt("Model",       session["model"])
            session["api_host"]    = _prompt("FastAPI URL", session["api_host"])
            ok("Configuration updated.")
            continue

        if choice in MENU:
            await _run_test(choice, session)
            _prompt("\nPress Enter to return to menu")
        else:
            print("  Invalid choice, try again.")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Ollama integration tests for DDoS Detection API")
    parser.add_argument(
        "--test",
        choices=["ping", "raw", "scenario", "analysis", "simulate", "edge", "all", "chat"],
        default="chat",
        help="Which test to run (default: chat — interactive menu)"
    )
    parser.add_argument("--host",   default=DEFAULT_OLLAMA_HOST, help="Ollama server URL")
    parser.add_argument("--api",    default=DEFAULT_API_HOST,    help="FastAPI server URL")
    parser.add_argument("--model",  default="llama3.1:8b",       help="Ollama model name")
    parser.add_argument("--attack", default=DEFAULT_ATTACK_TYPE, choices=CLASS_LABELS,
                        help="Attack type for tests (default: syn-flood)")
    args = parser.parse_args()

    if args.test == "chat":
        await interactive_mode(args.host, args.api, args.model)
        return

    # Batch mod — direktno pokretanje izabranog testa
    results = []

    if args.test in ("ping",     "all"): results.append(await test_ping(args.host))
    if args.test in ("raw",      "all"): results.append(await test_raw_generate(args.host, args.model))
    if args.test in ("scenario", "all"): results.append(await test_generate_scenario(args.host, args.model, args.attack))
    if args.test in ("analysis", "all"): results.append(await test_generate_analysis(args.host, args.model))
    if args.test in ("simulate", "all"): results.append(await test_simulate_endpoint(args.api, args.attack))
    if args.test in ("edge",     "all"): results.append(await test_edge_cases(args.api))

    if results:
        passed = sum(results)
        section(f"RESULT: {passed}/{len(results)} tests passed")
        sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())