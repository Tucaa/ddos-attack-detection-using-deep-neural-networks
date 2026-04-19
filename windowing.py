from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import random
import time
from functions import rand_uniform, rand_normal, clamp
import sys


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


# Karakteristike napada u sekundama
# Napomena - prob parametar je u procentima
ATTACK_PATTERN_DURATION = {
    "udp_large_packets": {
        "min_duration": 120,
        "max_duration": 1800,
        "typical_duration": 600,
        "long_attack_prob": 0.15,
        "long_duration_range": (1800, 3600),
    },
    "udp_flood_mixed": {
        "min_duration": 180,
        "max_duration": 2400,
        "typical_duration": 900,
        "long_attack_prob": 0.2,
        "long_duration_range": (1800, 3600),
    },
    "dns_amplification": {
        "min_duration": 300,
        "max_duration": 1800,
        "typical_duration": 600,
        "long_attack_prob": 0.10,
        "long_duration_range": (1800, 3600),
    },
    "ntp_amplification": {
        "min_duration": 240,
        "max_duration": 1500,
        "typical_duration": 480,
        "long_attack_prob": 0.08,
        "long_duration_range": (1500, 2700),
    },
    "ack_flood": {
        "min_duration": 1200,
        "max_duration": 5400,
        "typical_duration": 2700,
        "long_attack_prob": 0.30,
        "long_duration_range": (5400, 14400),
    },
    "icmp_flood": {
        "min_duration": 180,
        "max_duration": 1200,
        "typical_duration": 420,
        "long_attack_prob": 0.12,
        "long_duration_range": (1200, 2400),
    },
    "subnet_carpet_bombing": {
        "min_duration": 1800,
        "max_duration": 7200,
        "typical_duration": 3600,
        "long_attack_prob": 0.35,
        "long_duration_range": (7200, 21600),
    },
    "syn_flood": {
        "min_duration": 600,
        "max_duration": 3600,
        "typical_duration": 1800,
        "long_attack_prob": 0.25,
        "long_duration_range": (3600, 7200),
    },
    # Kod obicnog saobracaja se uzima
    "normal": {
        "min_duration": 3600,
        "max_duration": 86400,
        "typical_duration": 14400,
        "long_attack_prob": 0.0,
        "long_duration_range": (0, 0),
    },
}


def define_duration(attack_type: str) -> int:
    try:
        profile = ATTACK_PATTERN_DURATION.get(attack_type)

        if not profile:
            raise ValueError(f"Unknown parameter: {attack_type}. Known keys: {list(ATTACK_PATTERN_DURATION.keys())}")

        min_duration = profile["min_duration"]
        max_duration = profile["max_duration"]
        typical_duration = profile["typical_duration"]
        long_attack_prob = profile["long_attack_prob"]
        long_duration_range = profile["long_duration_range"]

        if random.random() < long_attack_prob:
            long_min, long_max = long_duration_range
            return int(rand_uniform(long_min, long_max))
        else:
            dev = (max_duration - min_duration) / 6.0
            duration = rand_normal(typical_duration, dev)
            return int(clamp(duration, min_duration, max_duration))
    except Exception as e:
        print(f'Exception windowing | define_duration: {e} Line: {sys.exc_info()[2].tb_lineno}')



# U buducnosti dodati talase napada posto moze i to da se desi
def waves(attack_type: str) -> bool:
    try:
        wave_probability = {
            "syn_flood": 0.3,
            "dns_amplification": 0.7,
            "ntp_amplification": 0.6,
            "udp_flood_mixed": 0.5,
            "icmp_flood": 0.4,
            "subnet-carpet_bombing": 0.6,
            "ack_flood": 0.2,
        }
        return random.random() < wave_probability.get(attack_type, 0.5)
    except Exception as e:
        print(f'Exception windowing | waves: {e} Line: {sys.exc_info()[2].tb_lineno}')


def wave_pattern(total_dur: int, wave_num: int) -> list[dict]:
    try:
        wave_dur = total_dur / wave_num
        # Inicijalno razlika imedju talasa 30% ako bude trebalo zameniti
        quiet_per = wave_dur * 0.3

        return [
            {
                "start_offset": int(i * wave_dur),
                "dur": int(wave_dur * 0.7),
                "wave_num": i + 1,
            }
            for i in range(wave_num)
        ]
    except Exception as e:
        print(f'Exception windowing | wave_pattern: {e} Line: {sys.exc_info()[2].tb_lineno}')


def generate_attack_windows(attack_fn, attack_type: str, start_timestamp: int, windows_ms: int) -> list[dict]:
    try:
        dur_sec = define_duration(attack_type)
        windows_num = max(1, int((dur_sec * 1000) / windows_ms))

        # Deo za talase napada
        multiple_waves = waves(attack_type)
        atk_wave = (
            wave_pattern(dur_sec, int(rand_uniform(2, 4)))
            if multiple_waves
            else [{"start_offset": 0, "dur": dur_sec, "wave_num": 1}]
        )

        result = []
        for i in range(windows_num):
            time_sec = (i * windows_ms) / 1000
            timestamp = start_timestamp + i * windows_ms

            in_wave = any(
                w["start_offset"] <= time_sec < w["start_offset"] + w["dur"]
                for w in atk_wave
            )
            active_wave = in_wave and random.random() > 0.1

            result.append(add_window_metadata(attack_fn(), i, timestamp, active_wave))

        return result
    except Exception as e:
        print(f'Exception windowing | generate_attack_windows: {e} Line: {sys.exc_info()[2].tb_lineno}')

# Izmenjeno je da se ova funkcija odnosi na normalni saobracaj
def generate_windows_normal(attack_fn, n: int, start_timestamp: int, window_ms: int) -> list[dict]:
    try:
        return [
            add_window_metadata(
                attack_fn(),
                i,
                start_timestamp + i * window_ms,
                # Za normalni saobracaj se uzima da je uvek aktivan
                1,
            )
            for i in range(n)
        ]
    except Exception as e:
        print(f'Exception windowing | generate_windows_normal: {e} Line: {sys.exc_info()[2].tb_lineno}')

# Funkcija za generisanje vektora sa svim opsezima instanci napada
def generate_attack_vector(attack_fn, attack_type: str, instances: int, duration_hours: float, start_timestamp: int, window_ms: int) -> list[dict]:
    try:
        duration_ms = duration_hours * 3600 * 1000
        spacing_ms = duration_ms / instances

        result = []
        for ind_instance in range(instances):
            instance_start = int(start_timestamp + ind_instance * spacing_ms)
            windows = generate_attack_windows(attack_fn, attack_type, instance_start, window_ms)

            for w in windows:
                result.append({
                    **w,
                    "instance_id": ind_instance,
                    "vector_id": f"{attack_type}-vector",
                })

        return result
    except Exception as e:
        print(f'Exception windowing | generate_attack_vector: {e} Line: {sys.exc_info()[2].tb_lineno}')

# Funkcija za generisanja timeline odnosno organizovanog dataseta
# kod kojeg se napadi ne preklapaju
def generate_timeline(window_ms: int, *attack_specs) -> list[dict]:

    try:
        start_ts = int(time.time() * 1000)
        curr_time = start_ts
        all_windows = []

        for spec_type, spec_data in attack_specs:
            if spec_type == "attack":
                new_window = generate_attack_windows(
                    spec_data["attack_fn"],
                    spec_data["attack_type"],
                    curr_time,
                    window_ms,
                )
            elif spec_type == "vector":
                new_window = generate_attack_vector(
                    spec_data["attack_fn"],
                    spec_data["attack_type"],
                    spec_data["instances"],
                    spec_data["duration_hours"],
                    curr_time,
                    window_ms,
                )
            elif spec_type == "normal":
                new_window = generate_windows_normal(
                    spec_data["normal_fn"],
                    spec_data["num_windows"],
                    curr_time,
                    window_ms,
                )
            else:
                raise ValueError(f"Unknown spec type: {spec_type}")

            last_ts = new_window[-1]["timestamp"] if new_window else curr_time

            # Za pocetak je stavljeno da miran period bude od 5 do 15 min
            # kasnije promeniti
            quiet_period_ms = int(rand_uniform(5, 15) * 60 * 1000)
            curr_time = last_ts + window_ms + quiet_period_ms

            all_windows.extend(new_window)

        return all_windows
    except Exception as e:
        print(f'Exception windowing | generate_timeline: {e} Line: {sys.exc_info()[2].tb_lineno}')