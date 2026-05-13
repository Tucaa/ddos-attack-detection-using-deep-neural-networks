"""
Salje CSV fajl na /predict/file endpoint i ispisuje rezultate.
Kasnije ces namestiti da se sve radi direkno preko api poziva ovo je inicijalno za testiranje
Upotreba:
    python inference_script.py traffic.csv
    python inference_script.py traffic.csv --url http://localhost:8000
    python inference_script.py traffic.csv --json
"""

import argparse
import json
import sys
import requests


def inference_script():
    try:
        parser = argparse.ArgumentParser(description="DDoS Detection — batch inference from CSV fajla")
        parser.add_argument("file",               help="Path to CSV")
        parser.add_argument("--url", default="http://localhost:8000")
        parser.add_argument("--json", action="store_true", help="Ispisi raw JSON odgovor")
        args = parser.parse_args()

        print(f"Sending: {args.file} -> {args.url}/predict/file")

        try:
            with open(args.file, "rb") as f:
                resp = requests.post(
                    f"{args.url}/predict/file",
                    files={"file": (args.file, f, "text/csv")},
                    timeout=120,
                )
            resp.raise_for_status()
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)
        except requests.HTTPError as e:
            print(f"HTTP error {e.response.status_code}: {e.response.text}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

        data = resp.json()

        if args.json:
            print(json.dumps(data, indent=2))
            return

        print(f"\nTotal windows:  {data['total_windows']}")
        print(f"Attack windows: {data['attack_windows']}\n")

        for p in data["predictions"]:
            status = "ATTACK" if p["is_attack"] else "normal"
            print(f"  Window {p['window_index']:>4}: {p['predicted_class']:<25} conf={p['confidence']:.3f}  [{status}]")
    except Exception as e:
        print(f'Exception api | inference_script: {e} Line: {sys.exc_info()[2].tb_lineno}')
      


if __name__ == "__main__":
    inference_script()