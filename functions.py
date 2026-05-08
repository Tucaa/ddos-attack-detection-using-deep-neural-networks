import random
import math
import csv
import torch
import sys
# from windowing import *
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Featurei koji su  u opsegu 0-1
RATIO_FEATURES = {
    "udp_ratio", "tcp_ratio", "icmp_ratio",
    "tcp_syn_ratio", "tcp_ack_ratio", "tcp_fin_ratio",
    "dns_query_ratio", "dns_response_ratio",
    "top_src_ip_packet_share", "top_src_ip_byte_share",
    "top_dst_port_share", "dst_subnet_spread",
}

# Featurei koji moraju biti pozitivni
POSITIVE_FEATURES = {
    "packet_rate", "byte_rate", "avg_packet_size", "std_packet_size",
    "unique_src_ips", "unique_dst_ips", "unique_flows",
    "src_ip_entropy", "dst_ip_entropy", "dst_port_entropy",
}

def rand_uniform(min_val: float, max_val: float) -> float:
    return min_val + random.random() * (max_val - min_val)

# Box-Muller transformation
def rand_normal(mean: float, std: float) -> float:
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mean + std * z

def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))

def write_csv(dataset: list[dict], filename: str) -> None:
    if not dataset:
        print("Dataset is empty, nothing to write.")
        return

    if not filename.endswith(".csv"):
        filename += ".csv"

    fieldnames = list(dataset[0].keys())

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

    print(f"Written {len(dataset)} rows to '{filename}'.")

# Iz windowinga prebaceno ovde!
def format_timestamp(ms: int) -> str:
    try:
        # Konvertuje milisekunde u formatirani string
        dt = datetime.fromtimestamp(ms / 1000)
        dt_final = dt.replace(tzinfo = ZoneInfo('UTC'))
        # Eventualno dodaj 2 funkcije Prvu koja konvertuje string u datetimeobj(UTC) obrnuto formatira dt. obj u string(u lokalnom vremenus)
        return dt_final.strftime("%Y-%m-%dT%H:%M:%S")
    
    except Exception as e:
        print(f'Exception windowing | format_timestamp: {e} Line: {sys.exc_info()[2].tb_lineno}')



# F-je koje dodaju vremenske serije podacima
def add_window_metadata(sample: dict, window_id: int, timestamp: int, active_atk: int) -> dict:
    return {
        **sample,
        "window_id": window_id,
        "timestamp": timestamp,
        "ts_formated": format_timestamp(timestamp),
        "attack_active": int(active_atk),
    }

# Iz generatte_dataset prebaceno ovde!
# Class weights helper funkcijaa za generisanje datasetta
def compute_class_weights(dataset: list[dict], labels: list[str]) -> dict:
    
    """
    Računa inverse-frequency weight po klasi.
    Vraća rečnik {label: weight} koji se prosleđuje CrossEntropyLoss-u.
    """

    try:
        counts = {label: 0 for label in labels}
        for sample in dataset:
            lbl = sample.get("label")
            if lbl in counts:
                counts[lbl] += 1

        total = sum(counts.values())
        n_classes = len(labels)

        weights = {}
        for label, count in counts.items():
            # Inverse frequency — klase sa manje uzoraka dobijaju veći weight
            weights[label] = total / (n_classes * count) if count > 0 else 1.0

        print("\nClass weights:")
        for label, w in weights.items():
            print(f"  {label:<25} count: {counts[label]:>7}   weight: {w:.4f}")

        return weights

    except Exception as e:
        print(f'Exception functions | compute_class_weights: {e} Line: {sys.exc_info()[2].tb_lineno}')


def get_class_weights_tensor(dataset: list[dict], labels: list[str], device) -> "torch.Tensor":
    try:
        weights_dict = compute_class_weights(dataset, labels)
        # Redosled mora da odgovara LabelEncoder.classes_
        weights_list = [weights_dict.get(label, 1.0) for label in labels]
        return torch.tensor(weights_list, dtype=torch.float32).to(device)
    except Exception as e:
        print(f'Exception functions | get_class_weights_tesnor: {e} Line: {sys.exc_info()[2].tb_lineno}')


def oversample_minority_classes(dataset: list[dict], labels: list[str], target_ratio: float = 0.5,) -> list[dict]:
    """
    Oversampluje manjinske klase do target_ratio * count(normal).
    target_ratio=0.5 znaci da svaka napadna klasa ima bar 50% uzoraka normal klase.
    Koristi add_noise za male varijacije pri dupliranju uzoraka.
    """
    try:
        # Grupisanje po labeli
        by_label = defaultdict(list)
        for sample in dataset:
            by_label[sample["label"]].append(sample)

        normal_count = len(by_label.get("normal", []))
        target_count = int(normal_count * target_ratio)

        print(f"\nOversampling — target po klasi: {target_count} (normal: {normal_count})")

        oversampled = list(dataset)

        for label in labels:
            if label == "normal":
                continue

            current = by_label.get(label, [])
            current_count = len(current)

            if current_count >= target_count:
                print(f"  {label:<25} {current_count:>7} — preskočeno")
                continue

            needed = target_count - current_count
            print(f"  {label:<25} {current_count:>7} → dodajem {needed} uzoraka")

            for i in range(needed):
                # Uzimamo nasumičan postojeći uzorak i dodajemo šum
                base = random.choice(current)
                noisy = add_noise(base, noise_level=0.04)
                # Čuvamo meta kolone nepromenjene osim timestamp-a
                noisy["label"]          = label
                noisy["attack_active"]  = base["attack_active"]
                noisy["window_id"]      = base["window_id"]
                noisy["timestamp"]      = base["timestamp"] + i  # mali offset
                noisy["ts_formated"]    = format_timestamp(noisy["timestamp"])
                oversampled.append(noisy)

        return oversampled
    except Exception as e:
        print(f'Exception functions | oversample_minority_classes: {e} Line: {sys.exc_info()[2].tb_lineno}')


def add_noise(sample: dict, noise_level: float = 0.05) -> dict:
    """
    Dodaje Gaussov sum na numericke featuere.
    noise_level = standardna devijacija kao procenat vrednosti featura.
    """
    try:
        noisy = dict(sample)
        for key, val in sample.items():
            if not isinstance(val, (int, float)):
                continue
            if key in ("window_id", "timestamp", "attack_active"):
                continue

            noise = np.random.normal(0, abs(val) * noise_level + 1e-6)
            noisy_val = val + noise

            if key in RATIO_FEATURES:
                noisy_val = float(np.clip(noisy_val, 0.0, 1.0))
            elif key in POSITIVE_FEATURES:
                noisy_val = max(0.0, noisy_val)

            noisy[key] = noisy_val

        return noisy
    except Exception as e:
        print(f'Exception functions | add_noise: {e} Line: {sys.exc_info()[2].tb_lineno}')

def blend_samples(sample_a: dict, sample_b: dict, alpha: float) -> dict:
    """
    Linearno interpoluje izmedju dva uzorka.
    alpha=0.0 > sample_a, alpha=1.0 > sample_b
    Koristi se za tranzicione periode.
    """
    try:
        blended = dict(sample_a)
        for key, val_a in sample_a.items():
            if not isinstance(val_a, (int, float)):
                continue
            if key in ("window_id", "timestamp", "attack_active", "label"):
                continue
            val_b = sample_b.get(key, val_a)
            blended[key] = val_a * (1 - alpha) + val_b * alpha

        return blended
    except Exception as e:
        print(f'Exception dataset_generattor | blend_samples: {e} Line: {sys.exc_info()[2].tb_lineno}')

def generate_transition(from_fn,to_fn, from_label: str, to_label: str, n_windows: int, start_ts: int, window_ms: int,) -> list[dict]:
    """
    Generise tranzicioni period između dva tipa saobracaja.
    Prvih 50% prozora: label from_label sa rastucim alpha ka to_fn
    Drugih 50% prozora: label to_label sa opadajucim alpha
    """
    try:
        result = []
        for i in range(n_windows):
            alpha = i / n_windows  # 0.0 → 1.0
            base_a = from_fn()
            base_b = to_fn()

            blended = blend_samples(base_a, base_b, alpha)
            blended = add_noise(blended, noise_level=0.08)

            # Labela prati dominantni tip
            label = from_label if alpha < 0.5 else to_label
            ts = start_ts + i * window_ms

            result.append({
                **blended,
                "label": label,
                "window_id": i,
                "timestamp": ts,
                "ts_formated": format_timestamp(ts),
                "attack_active": 1 if label != "normal" else 0,
            })

        return result
    except Exception as e:
        print(f'Exception functions | generate_transition: {e} Line: {sys.exc_info()[2].tb_lineno}')