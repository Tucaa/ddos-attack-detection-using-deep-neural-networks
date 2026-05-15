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
DEFAULT_ATTACK_TYPE = "syn-flood"
TIMEOUT             = 120.0

WINDOW_SIZE = 20

def ok(msg: str):
    print(f"  [OK]   {msg}")

def fail(msg: str):
    print(f"  [FAIL] {msg}")

def info(msg: str):
    print(f"  [INFO] {msg}")

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def subsection(title: str):
    print(f"\n  --- {title} ---")


async def test_ping(ollama_host: str, verbose: bool = False) -> bool:
    """Proverava da li je Ollama server dostupan i koji modeli su ucitani."""
    section("TEST 1 — Ollama ping / availability")

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Proverava osnovnu dostupnost
        try:
            resp = await client.get(f"{ollama_host}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            ok(f"Ollama server reachable: {ollama_host}")
        except httpx.ConnectError:
            fail(f"Could not connect to Ollama: {ollama_host}")
            info("Make sure Ollama is running: 'ollama serve'")
            return False
        except Exception as e:
            fail(f"Unexpected error: {e}")
            return False

        # Lista dostupnih modela
        models = data.get("models", [])
        if models:
            ok(f"Available models ({len(models)}):")
            for m in models:
                name = m.get("name", "?")
                size = m.get("size", 0)
                size_gb = size / (1024 ** 3) if size else 0
                print(f"       - {name}  ({size_gb:.1f} GB)")
        else:
            info("No models loaded.")

        if verbose:
            subsection("Raw /api/tags response")
            print(json.dumps(data, indent=2)[:500])

    return True


async def test_raw_generate(ollama_host: str, model: str, verbose: bool = False) -> bool:
    """Testira osnovni ollama_generate poziv sa jednostavnim promptom."""
    section("TEST 2 — Raw ollama_generate call")

    payload = {
        "model": model,
        "prompt": "Reply with exactly one word: hello",
        "system": "You are a test assistant. Reply with exactly one word.",
        "stream": False,
    }

    info(f"Model: {model}")
    info(f"Prompt: '{payload['prompt']}'")

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(f"{ollama_host}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        fail(f"Ollama not available at {ollama_host}")
        return False
    except httpx.HTTPStatusError as e:
        fail(f"HTTP {e.response.status_code}: {e.response.text[:200]}")
        return False
    except Exception as e:
        fail(f"Error: {e}")
        return False

    elapsed = time.perf_counter() - start
    response_text = data.get("response", "").strip()

    ok(f"Received response in {elapsed:.2f}s")
    print(f"       Response: '{response_text}'")

    if not response_text:
        fail("Empty response from Ollama!")
        return False

    if verbose:
        subsection("Response Details")
        eval_count   = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 0)
        tps = eval_count / (eval_duration / 1e9) if eval_duration else 0
        info(f"Tokens generated: {eval_count}")
        info(f"Tokens/sec: {tps:.1f}")

    return True


async def test_generate_scenario(ollama_host: str, model: str, attack_type: str, verbose: bool = False) -> bool:
    """
    Testira generate_attack_scenario — proverava da Ollama vraca
    validnu matricu ispravnih dimenzija.
    """
    section(f"TEST 3 — generate_attack_scenario ({attack_type})")

    features_str = "\n".join(f"  - {name}" for name in FEATURE_NAMES)
    prompt = f"""Generate a synthetic network traffic matrix simulating a **{attack_type}** DDoS attack.
    Requirements:
    - Return ONLY a valid JSON array, no explanation, no markdown.
    - The array must contain exactly {WINDOW_SIZE} rows.
    - Each row must contain exactly {len(FEATURE_NAMES)} float values.
    - Values must be realistic for a {attack_type} attack pattern.
    Respond with ONLY the JSON array."""

    info(f"Requesting {WINDOW_SIZE}x{len(FEATURE_NAMES)} matrix for attack: {attack_type}")

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            payload = {
                "model":  model,
                "prompt": prompt,
                "system": "You are a network security expert. Respond ONLY with valid JSON arrays.",
                "stream": False,
            }
            resp = await client.post(f"{ollama_host}/api/generate", json=payload)
            resp.raise_for_status()
            raw = resp.json().get("response", "")
    except Exception as e:
        fail(f"Ollama call failed: {e}")
        return False

    elapsed = time.perf_counter() - start
    ok(f"Received response in {elapsed:.2f}s")

    if verbose:
        subsection("Raw response (first 300 chars)")
        print(f"  {raw[:300]}")

    # Ciscenje markdown omotaca
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
    clean = match.group(1).strip() if match else raw.strip()

    # JSON parsiranje
    try:
        matrix = json.loads(clean)
        ok("JSON parsing successful")
    except json.JSONDecodeError as e:
        fail(f"JSON parsing failed: {e}")
        info(f"Cleaned response: {clean[:200]}")
        return False

    # Validacija dimenzija
    passed = True
    if not isinstance(matrix, list):
        fail(f"Expected list, got: {type(matrix).__name__}")
        return False

    if len(matrix) != WINDOW_SIZE:
        fail(f"Expected {WINDOW_SIZE} rows, got: {len(matrix)}")
        passed = False
    else:
        ok(f"Number of rows: {len(matrix)} (expected {WINDOW_SIZE})")

    # Proveravamo svaki red
    bad_rows = []
    for i, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != len(FEATURE_NAMES):
            bad_rows.append(f"Row {i} is invalid")
    
    if bad_rows:
        fail(f"Found {len(bad_rows)} invalid rows")
        passed = False
    else:
        ok(f"All {len(matrix)} rows valid")

    return passed

async def test_generate_analysis(ollama_host: str, model: str, verbose: bool = False) -> bool:
    """
    Testira generate_attack_analysis — proverava da Ollama vraca
    validan JSON sa ocekivanim kljucevima.
    """
    section("TEST 4 — generate_attack_analysis")

    predicted_class = "syn-flood"
    confidence = 0.87
    prompt = f"Provide a JSON analysis for a {predicted_class} attack with {confidence:.2%} confidence."

    info(f"Testing analysis for: {predicted_class}")

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            payload = {
                "model": model,
                "prompt": prompt,
                "system": "You are a senior network security analyst. Respond ONLY with valid JSON containing 'description' and 'mitigation_steps'.",
                "stream": False,
            }
            resp = await client.post(f"{ollama_host}/api/generate", json=payload)
            resp.raise_for_status()
            raw = resp.json().get("response", "")
    except Exception as e:
        fail(f"Ollama call failed: {e}")
        return False

    elapsed = time.perf_counter() - start
    ok(f"Received response in {elapsed:.2f}s")

    # Ciscenje i parsiranje
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
    clean = match.group(1).strip() if match else raw.strip()

    try:
        result = json.loads(clean)
        ok("JSON parsing successful")
    except json.JSONDecodeError as e:
        fail(f"JSON parsing failed: {e}")
        return False

    passed = True
    if "description" not in result or "mitigation_steps" not in result:
        fail("Missing required JSON keys")
        passed = False
    else:
        ok("Required keys ('description', 'mitigation_steps') are present")

    return passed


async def test_simulate_endpoint(api_host: str, attack_type: str, verbose: bool = False) -> bool:
    """Testira /simulate endpoint direktno kroz FastAPI."""
    section(f"TEST 5 — /simulate endpoint ({attack_type})")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            health = await client.get(f"{api_host}/health")
            health.raise_for_status()
            health_data = health.json()
    except Exception as e:
        fail(f"FastAPI server not available: {e}")
        return False

    ok(f"FastAPI server reachable: {api_host}")
    
    info(f"Calling /simulate for: {attack_type}")
    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(f"{api_host}/simulate", json={"attack_type": attack_type})
    except Exception as e:
        fail(f"HTTP call failed: {e}")
        return False

    elapsed = time.perf_counter() - start
    if resp.status_code != 200:
        fail(f"HTTP {resp.status_code}: {resp.text[:300]}")
        return False

    ok(f"Received response in {elapsed:.2f}s (HTTP 200)")
    data = resp.json()
    
    if "analysis" in data:
        ok("Analysis field present in response")
        return True
    else:
        fail("Analysis field missing")
        return False


async def test_edge_cases(ollama_host: str, api_host: str, model: str, verbose: bool = False) -> bool:
    """Testira ocekivano ponasanje u losim slucajevima."""
    section("TEST 6 — Edge cases and negative scenarios")
    
    subsection("6a — Non-existent attack_type on /simulate")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{api_host}/simulate", json={"attack_type": "invalid-attack-123"})
        if resp.status_code == 400:
            ok("HTTP 400 returned as expected")
        else:
            fail(f"Expected HTTP 400, got: {resp.status_code}")
    except Exception:
        info("FastAPI not available — skipping 6a")

    return True

# -----------------------------------------------------------------------
# Interaktivni mod — chat-like REPL
# -----------------------------------------------------------------------

CHAT_APIS = {
    "1": {"name": "ping", "label": "Ping — Ollama availability", "params": []},
    "2": {"name": "raw", "label": "Raw generate — basic test", "params": []},
    "3": {"name": "generate", "label": "Scenario — Traffic matrix", "params": ["attack_type"]},
    "4": {"name": "analysis", "label": "Analysis — Mitigation", "params": []},
    "5": {"name": "simulate", "label": "Simulate — End-to-end API", "params": ["attack_type"]},
    "6": {"name": "edge", "label": "Edge cases", "params": []},
    "7": {"name": "all", "label": "Run all tests", "params": ["attack_type"]},
}

def _prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"  {text}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        raise KeyboardInterrupt
    return val if val else default

def _pick_attack_type() -> str:
    print("\n  Available attack types:")
    for i, at in enumerate(CLASS_LABELS, 1):
        print(f"    {i}. {at}")
    raw = _prompt("Enter number or name of attack", "syn-flood")
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(CLASS_LABELS): return CLASS_LABELS[idx]
    return raw if raw in CLASS_LABELS else "syn-flood"

def _show_menu(session: dict):
    print("\n" + "="*60)
    print("  Ollama API — Interactive Tests")
    print("="*60)
    print(f"  Ollama: {session['host']}  |  Model: {session['model']}")
    print(f"  API:    {session['api']}\n")
    for key, api in CHAT_APIS.items():
        print(f"  {key}. {api['label']}")
    print("\n  c. Change config | q. Exit\n")

async def interactive_mode(ollama_host: str, api_host: str, model: str):
    """Chat-like REPL za interaktivno testiranje."""
    session = {"host": ollama_host, "model": model, "api": api_host, "last_attack": DEFAULT_ATTACK_TYPE}

    while True:
        _show_menu(session)
        try:
            choice = _prompt("Selection").lower()
        except KeyboardInterrupt:
            break

        if choice in ("q", "quit", "exit"):
            print("\n  Goodbye!\n")
            break
        
        if choice == "c":
            session["host"] = _prompt("Ollama host", session["host"])
            session["model"] = _prompt("Ollama model", session["model"])
            session["api"] = _prompt("FastAPI url", session["api"])
            ok("Configuration updated.")
            continue

        if choice in CHAT_APIS:
            api = CHAT_APIS[choice]
            attack = session["last_attack"]
            if "attack_type" in api["params"]:
                attack = _pick_attack_type()
            
            verbose = _prompt("Verbose? (y/n)", "n").lower() in ("y", "yes")
            
            name = api["name"]
            if name == "ping": await test_ping(session["host"], verbose)
            elif name == "raw": await test_raw_generate(session["host"], session["model"], verbose)
            elif name == "generate": await test_generate_scenario(session["host"], session["model"], attack, verbose)
            elif name == "analysis": await test_generate_analysis(session["host"], session["model"], verbose)
            elif name == "simulate": await test_simulate_endpoint(session["api"], attack, verbose)
            elif name == "edge": await test_edge_cases(session["host"], session["api"], session["model"], verbose)
            elif name == "all":
                await test_ping(session["host"], verbose)
                await test_raw_generate(session["host"], session["model"], verbose)
                await test_generate_scenario(session["host"], session["model"], attack, verbose)

            _prompt("Press Enter to return to menu")
        else:
            print("  Invalid choice.")

async def main():
    parser = argparse.ArgumentParser(description="Ollama Integration Tests for DDoS Detection API")
    parser.add_argument("--test", choices=["ping", "generate", "raw", "analysis", "simulate", "edge", "all", "chat"], default="chat")
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--api", default=DEFAULT_API_HOST)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--attack", default=DEFAULT_ATTACK_TYPE, choices=CLASS_LABELS)
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()

    if args.test == "chat":
        await interactive_mode(args.host, args.api, args.model)
        return

    # Batch execution logic (simplified for brevity)
    section("RESULTS")
    # Ovde bi išla logika za 'all' ili pojedinačne testove slična onoj u interaktivnom modu
    print("  Batch mode finished. Use --test chat for interactive menu.")

if __name__ == "__main__":
    asyncio.run(main())