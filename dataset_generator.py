from attacks import *
from normal import *
from windowing import *
from functions import *
from functions import write_csv
import random
import time
import sys




# Featurei koji su  u opsegu 0-1
# RATIO_FEATURES = {
#     "udp_ratio", "tcp_ratio", "icmp_ratio",
#     "tcp_syn_ratio", "tcp_ack_ratio", "tcp_fin_ratio",
#     "dns_query_ratio", "dns_response_ratio",
#     "top_src_ip_packet_share", "top_src_ip_byte_share",
#     "top_dst_port_share", "dst_subnet_spread",
# }

# # Featurei koji moraju biti pozitivni
# POSITIVE_FEATURES = {
#     "packet_rate", "byte_rate", "avg_packet_size", "std_packet_size",
#     "unique_src_ips", "unique_dst_ips", "unique_flows",
#     "src_ip_entropy", "dst_ip_entropy", "dst_port_entropy",
# }


ALL_ATTACK_CONFIGS = [
    ("syn_flood",             syn_flood),
    ("dns_amplification",     dns_amplification),
    ("subnet_carpet_bombing", subnet_carpet_bombing),
    ("udp_flood_large",       udp_flood_large),
    ("icmp_flood",            icmp_flood),
    ("udp_flood_mixed",       udp_flood_mixed),
    ("ntp_amplification",     ntp_amplification),
    ("ack_flood",             ack_flood),
]

# Koliko čistih uzoraka po klasi po iteraciji
SAMPLES_PER_CLASS  = 150
# Koliko tranzicionih prozora na ulazu/izlazu napada
TRANSITION_WINDOWS = 20


# Nova implementacija sa oversamplovanjem
def generate_mixed_dataset(window_ms: int, days: int) -> list[dict]:

    try:
        start_ts = int(time.time() * 1000)
        day_ms   = 24 * 60 * 60 * 1000
        total_ms = days * day_ms
        # Krajnji timestamp za proveru
        end_ts   = start_ts + total_ms

        curr_ts     = start_ts
        all_windows = []

        while curr_ts < end_ts:

            # Nasumičan redosled klasa za svaku iteraciju!! (po potrebi promeniti)
            shuffled_attacks = ALL_ATTACK_CONFIGS.copy()
            random.shuffle(shuffled_attacks)

            #TTimeline svi napadi su u nasumičnom redosledu
            timeline_specs = []
            for attack_type, attack_fn in shuffled_attacks:
                timeline_specs.append(
                    ("attack", {"attack_fn": attack_fn, "attack_type": attack_type})
                )
                # Normalan saobraćaj između svakog napada
                timeline_specs.append(
                    ("normal", {"normal_fn": normal_mixed_traffic, "num_windows": 80})
                )

            timeline_attacks = generate_timeline(window_ms, *timeline_specs)

            # print('TIMELINE NAPADI', timeline_attacks)

            # Dodaje se šum na timeline uzorke
            timeline_attacks = [
                {
                    **add_noise(s, noise_level=0.05),
                    "label":          s["label"],
                    "timestamp":      s["timestamp"],
                    "ts_formated":    s["ts_formated"],
                    "attack_active":  s["attack_active"],
                    "window_id":      s["window_id"],
                }
                for s in timeline_attacks
            ]

            # Balansirani uzorci sa tranzicijama
            # Nasumičan redosled i ovde (isti shuffled_attacks)
            balanced_samples = []
            for class_idx, (attack_type, attack_fn) in enumerate(shuffled_attacks):
                offset = class_idx * (SAMPLES_PER_CLASS + TRANSITION_WINDOWS * 2)

                # Tranzicija: normal > napad
                trans_in = generate_transition(
                    from_fn=normal_mixed_traffic,
                    to_fn=attack_fn,
                    from_label="normal",
                    to_label=attack_type,
                    n_windows=TRANSITION_WINDOWS,
                    start_ts=curr_ts + offset * window_ms,
                    window_ms=window_ms,
                )

                # Čisti uzorci napada sa šumom
                clean_attack = []
                for idx in range(SAMPLES_PER_CLASS):
                    sample = add_noise(attack_fn(), noise_level=0.05)
                    ts = curr_ts + (offset + TRANSITION_WINDOWS + idx) * window_ms
                    clean_attack.append({
                        **sample,
                        "label":         attack_type,
                        "window_id":     idx,
                        "timestamp":     ts,
                        "ts_formated":   format_timestamp(ts),
                        "attack_active": 1,
                    })

                # Tranzicija: napad > normal
                trans_out = generate_transition(
                    from_fn=attack_fn,
                    to_fn=normal_mixed_traffic,
                    from_label=attack_type,
                    to_label="normal",
                    n_windows=TRANSITION_WINDOWS,
                    start_ts=curr_ts + (offset + TRANSITION_WINDOWS + SAMPLES_PER_CLASS) * window_ms,
                    window_ms=window_ms,
                )

                balanced_samples.extend(trans_in)
                balanced_samples.extend(clean_attack)
                balanced_samples.extend(trans_out)

            #Normalan saobraćaj
            last_timeline_ts = timeline_attacks[-1]["timestamp"] if timeline_attacks else curr_ts

            extra_normal_raw = generate_windows_normal(normal_mixed_traffic, 1000, curr_ts, window_ms)
            extra_normal = [
                {
                    **add_noise(s, noise_level=0.05),
                    "label":         s["label"],
                    "timestamp":     s["timestamp"],
                    "ts_formated":   s["ts_formated"],
                    "attack_active": s["attack_active"],
                    "window_id":     s["window_id"],
                }
                for s in extra_normal_raw
            ]

            last_ts = extra_normal[-1]["timestamp"] if extra_normal else last_timeline_ts

            all_windows.extend(timeline_attacks)
            all_windows.extend(balanced_samples)
            all_windows.extend(extra_normal)

            curr_ts = last_ts + window_ms

        # Oversampling manje zasutpoljenih klasa
        all_windows = oversample_minority_classes(
            all_windows,
            labels=[label for label, _ in [("normal", None)] + ALL_ATTACK_CONFIGS],
            target_ratio=0.5,
        )

        return all_windows
    
    except Exception as e:
        print(f'Exception dataset_generator | generate_mixed_dataset: {e} Line: {sys.exc_info()[2].tb_lineno}')

        






# Stara implementacija 
def generate_mixed_dataset_old(window_ms: int, days: int) -> list[dict]:
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
            ("udp_flood_large",   udp_flood_large),
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