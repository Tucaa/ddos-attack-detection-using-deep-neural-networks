import random
import math
import csv


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