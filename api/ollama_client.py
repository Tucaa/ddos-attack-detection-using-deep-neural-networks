import json
import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
TIMEOUT = 120.0


def _extract_json(raw: str) -> str:
    """
    Izvlaci JSON iz odgovora koji moze biti umotan u markdown code blok.
    Ollama cesto vraca ```json ... ``` omotac.
    """
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
    if match:
        return match.group(1).strip()
    return raw.strip()


async def ollama_generate(prompt: str, system: str = "") -> str:
    """
    Genericki poziv Ollama /api/generate endpointa.
    Vraca sirovi tekstualni odgovor modela.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            response = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except httpx.HTTPError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise RuntimeError(f"Ollama not available: {e}")


async def generate_attack_scenario(attack_type: str, window_size: int,feature_names: list[str]) -> list[list[float]]:
    """
    Koristi Ollamu da generise matricu mreznog saobracaja koja simulira
    odredjeni tip DDoS napada.
    Vraca listu od window_size uzoraka, svaki sa len(feature_names) float vrednosti.
    """
    features_str = "\n".join(
        f"  - {name}" for name in feature_names
    )

    prompt = f"""Generate a synthetic network traffic matrix simulating a **{attack_type}** DDoS attack.

Requirements:
- Return ONLY a valid JSON array, no explanation, no markdown.
- The array must contain exactly {window_size} rows.
- Each row must contain exactly {len(feature_names)} float values.
- Values must be realistic for a {attack_type} attack pattern.
- All values must be non-negative floats.

Features (in order):
{features_str}

Respond with ONLY the JSON array. Example format:
[[0.1, 0.5, 1.2, ...], [0.2, 0.4, 1.1, ...], ...]"""

    system = (
        "You are a network security expert. "
        "You generate synthetic network traffic data for DDoS simulation. "
        "You respond ONLY with valid JSON arrays. No explanations, no markdown."
    )

    raw = await ollama_generate(prompt, system)
    clean = _extract_json(raw)

    try:
        matrix = json.loads(clean)
    except json.JSONDecodeError as e:
        logger.error(f"Ollama returned invalid JSON for scenario: {e}\nRaw: {raw[:300]}")
        raise ValueError(f"Ollama did not return valid JSON: {e}")
    
    if not isinstance(matrix, list) or len(matrix) != window_size:
        raise ValueError(
            f"Expected {window_size} rows, got {len(matrix) if isinstance(matrix, list) else 'not a list'}"
        )

    # Normalizujemo na float i clamepujemo negativne vrednosti
    try:
        matrix = [[max(0.0, float(v)) for v in row] for row in matrix]
    except (TypeError, ValueError) as e:
        raise ValueError(f"Matrix contains non-numeric values: {e}")

    return matrix


async def generate_attack_analysis(predicted_class: str, confidence: float, is_attack: bool, class_probabilities: dict[str, float],attack_type_requested: str) -> dict:
    """
    Koristi Ollamu da analizira rezultate LSTM modela i generise:
    - detaljan opis detektovanog napada
    - konkretne instrukcije za mitigaciju
    Vraca dict sa kljucevima 'description' i 'mitigation_steps'.
    """
    probs_str = "\n".join(
        f"  - {cls}: {prob:.2%}"
        for cls, prob in sorted(class_probabilities.items(), key=lambda x: -x[1])
    )

    prompt = f"""A DDoS detection LSTM model analyzed network traffic and returned the following results:

            - Requested simulation: {attack_type_requested}
            - Predicted class: {predicted_class}
            - Confidence: {confidence:.2%}
            - Is attack: {is_attack}
            - Class probabilities:
            {probs_str}

            Provide a JSON response with exactly these two keys:
            1. "description": A technical paragraph (3-5 sentences) describing the characteristics of a {predicted_class} attack, typical behavior, and why the model classified it this way.
            2. "mitigation_steps": A list of 5-7 concrete, actionable mitigation steps a network administrator should take immediately.

            Respond ONLY with valid JSON. No markdown, no explanation outside the JSON."""

    system = (
        "You are a senior network security analyst. "
        "You provide precise, actionable DDoS attack analysis and mitigation guidance. "
        "You respond ONLY with valid JSON."
    )

    raw = await ollama_generate(prompt, system)
    clean = _extract_json(raw)

    try:
        result = json.loads(clean)
    except json.JSONDecodeError as e:
        logger.error(f"Ollama returned invalid JSON for analysis: {e}\nRaw: {raw[:300]}")
        # Fallback — ako JSON parsing ne uspe, vracamo sirovi tekst
        return {
            "description": raw[:500],
            "mitigation_steps": ["Look server log details."],
        }

    return {
        "description": result.get("description", ""),
        "mitigation_steps": result.get("mitigation_steps", []),
    }