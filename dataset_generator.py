from attacks import *
from normal import *
from windowing import *
from functions import write_csv
import numpy as np
import time


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


def add_noise(sample: dict, noise_level: float = 0.05) -> dict:
    """
    Dodaje Gaussov sum na numericke featuere.
    noise_level = standardna devijacija kao procenat vrednosti featura.
    """
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


def blend_samples(sample_a: dict, sample_b: dict, alpha: float) -> dict:
    """
    Linearno interpoluje izmedju dva uzorka.
    alpha=0.0 > sample_a, alpha=1.0 > sample_b
    Koristi se za tranzicione periode.
    """
    blended = dict(sample_a)
    for key, val_a in sample_a.items():
        if not isinstance(val_a, (int, float)):
            continue
        if key in ("window_id", "timestamp", "attack_active", "label"):
            continue
        val_b = sample_b.get(key, val_a)
        blended[key] = val_a * (1 - alpha) + val_b * alpha

    return blended


def generate_transition(from_fn,to_fn, from_label: str, to_label: str, n_windows: int, start_ts: int, window_ms: int,) -> list[dict]:
    """
    Generise tranzicioni period između dva tipa saobracaja.
    Prvih 50% prozora: label from_label sa rastucim alpha ka to_fn
    Drugih 50% prozora: label to_label sa opadajucim alpha
    """
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



def generate_mixed_dataset(window_ms: int, days: int) -> list[dict]:
    start_ts = int(time.time() * 1000)
    day_ms = 24 * 60 * 60 * 1000
    total_ms = days * day_ms
    # Krajnji timestamp za proveru
    end_ts = start_ts + total_ms

    curr_ts = start_ts
    all_windows = []

    while curr_ts < end_ts:


        timeline_attacks = generate_timeline(
            window_ms,
            ("attack", {"attack_fn": syn_flood,            "attack_type": "syn_flood"}),
            ("normal", {"normal_fn": normal_mixed_traffic, "num_windows": 100}),
            ("attack", {"attack_fn": dns_amplification,    "attack_type": "dns_amplification"}),
            ("normal", {"normal_fn": normal_mixed_traffic, "num_windows": 100}),
            ("attack", {"attack_fn": subnet_carpet_bombing,"attack_type": "subnet_carpet_bombing"}),
        )

        # Dodaj sum na sve timeline uzorke
        timeline_attacks = [
            {**add_noise(s, noise_level=0.05), 
             "label": s["label"],
             "timestamp": s["timestamp"],
             "ts_formated": s["ts_formated"],
             "attack_active": s["attack_active"],
             "window_id": s["window_id"]}
            for s in timeline_attacks
        ]


        # Normal -> napad i napad -> normal za svaku klasu
        attack_configs = [
            ("udp_flood_large",   udp_large_packets),
            ("icmp_flood",        icmp_flood),
            ("udp_flood_mixed",   udp_flood_mixed),
            ("ntp_amplification", ntp_amplification),
            ("ack_flood",         ack_flood),
        ]

        balanced_samples = []
        for class_idx, (attack_type, attack_fn) in enumerate(attack_configs):
            offset = class_idx * 140  # 100 cistih + 20 tranzicija sa obe strane

            # Prva Tranzicija: zmedju napada i normalnog saobracaja stavljeno je inicijalno 20 prozora po potrebi promeniti
            trans_in = generate_transition(
                from_fn=normal_mixed_traffic,
                to_fn=attack_fn,
                from_label="normal",
                to_label=attack_type,
                n_windows=20,
                start_ts=curr_ts + offset * window_ms,
                window_ms=window_ms,
            )

            # cisti uzorci napada sa sumom (100 prozora)
            clean_attack = []
            for idx in range(100):
                sample = add_noise(attack_fn(), noise_level=0.05)
                ts = curr_ts + (offset + 20 + idx) * window_ms
                clean_attack.append({
                    **sample,
                    "label": attack_type,
                    "window_id": idx,
                    "timestamp": ts,
                    "ts_formated": format_timestamp(ts),
                    "attack_active": 1,
                })

            # Ponovna Tranzicija: izmedju napada i normalnog saobracaja stavljeno je inicijalno 20 prozora po potrebi promeniti
            trans_out = generate_transition(
                from_fn=attack_fn,
                to_fn=normal_mixed_traffic,
                from_label=attack_type,
                to_label="normal",
                n_windows=20,
                start_ts=curr_ts + (offset + 120) * window_ms,
                window_ms=window_ms,
            )

            balanced_samples.extend(trans_in)
            balanced_samples.extend(clean_attack)
            balanced_samples.extend(trans_out)

        # Normalan saobracaj sa sumom
        last_timeline_ts = timeline_attacks[-1]["timestamp"] if timeline_attacks else curr_ts

        extra_normal_raw = generate_windows_normal(normal_mixed_traffic, 1000, curr_ts, window_ms)
        extra_normal = [
            {**add_noise(s, noise_level=0.05),
             "label": s["label"],
             "timestamp": s["timestamp"],
             "ts_formated": s["ts_formated"],
             "attack_active": s["attack_active"],
             "window_id": s["window_id"]}
            for s in extra_normal_raw
        ]

        last_ts = extra_normal[-1]["timestamp"] if extra_normal else last_timeline_ts

        all_windows.extend(timeline_attacks)
        all_windows.extend(balanced_samples)
        all_windows.extend(extra_normal)

        curr_ts = last_ts + window_ms

    return all_windows


def handle_generate():
    filename = input("Output file name: ").strip() or "output.csv"
    days_str = input("Number of days (timeperiod): ").strip()
    days = int(days_str)

    print("Generating dataset!")
    dataset = generate_mixed_dataset(5000, days)
    write_csv(dataset, filename)
    print("Finished!")



if __name__ == "__main__":
    # Kasnije dodaj jos argumenata
    handle_generate()