from attacks import *
from normal import *
from windowing import *
from functions import write_csv

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
            ("normal", {"normal_fn": normal_mixed_traffic,   "num_windows": 100}),
            ("attack", {"attack_fn": dns_amplification,     "attack_type": "dns_amplification"}),
            ("normal", {"normal_fn": normal_mixed_traffic,   "num_windows": 100}),
            ("attack", {"attack_fn": subnet_carpet_bombing, "attack_type": "subnet_carpet_bombing"}),
        )

        # Offset da se timestamps ne preklapaju izmedju klasa
        attack_configs = [
            ("udp_flood_large",        udp_large_packets),
            ("icmp_flood",             icmp_flood),
            ("udp_flood_mixed",        udp_flood_mixed),
            ("ntp_amplification",      ntp_amplification),
            ("ack_flood",              ack_flood),
        ]

        balanced_samples = []
        for class_idx, (attack_type, attack_fn) in enumerate(attack_configs):
            offset = class_idx * 100
            for idx in range(100):
                sample = attack_fn()
                ts = curr_ts + (offset + idx) * window_ms
                balanced_samples.append({
                    **sample,
                    "label": attack_type,
                    "window_id": idx,
                    # Svaki uzorak dobija razlicit timestamp
                    "timestamp": ts,
                    "ts_formated": format_timestamp(ts),
                    # Stavljeno je da su svi napadi aktivni
                    "attack_active": 1,
                })

        last_timeline_ts = timeline_attacks[-1]["timestamp"] if timeline_attacks else curr_ts

        extra_normal = generate_windows_normal(normal_mixed_traffic, 1000, curr_ts, window_ms)
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