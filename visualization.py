import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
# from datetime import datetime


# Boje po klasi

CLASS_COLORS = {
    "normal":                 "#2ecc71",
    "syn_flood":              "#e74c3c",
    "dns_amplification":      "#e67e22",
    "subnet_carpet_bombing":  "#9b59b6",
    "udp_flood_large":        "#3498db",
    "icmp_flood":             "#f39c12",
    "udp_flood_mixed":        "#1abc9c",
    "ntp_amplification":      "#e91e63",
    "ack_flood":              "#795548",
}


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["ts_formated"] = pd.to_datetime(df["ts_formated"])
    df = df.sort_values("ts_formated").reset_index(drop=True)
    return df


# Byte rate kroz vreme (bojeno po klasi)

def plot_traffic_timeline(
    df: pd.DataFrame,
    metric: str = "byte_rate",
    title: str = "Network traffic trought time",
    resample: str = "1min",
    save_path: str = None,
):
    """
    Linijski grafik metrike kroz vreme.
    Svaki segment je obojen prema klasi saobracaja.
    resample: '1min', '5min', '30min' - agregacija po vremenskom intervalu.
    """
    fig, ax = plt.subplots(figsize=(18, 6))

    # Resample - prosek metrike po vremenskom intervalu
    df_grouped = (
        df.set_index("ts_formated")
        .groupby([pd.Grouper(freq=resample), "label"])[metric]
        .mean()
        .reset_index()
    )


    for label, color in CLASS_COLORS.items():
        subset = df_grouped[df_grouped["label"] == label]
        if subset.empty:
            continue
        ax.scatter(
            subset["ts_formated"],
            subset[metric],
            color=color,
            s=6,
            alpha=0.7,
            label=label,
            zorder=2,
        )

    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel(_metric_label(metric), fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.xticks(rotation=35)

    ax.legend(loc="upper left", fontsize=8, markerscale=2.5, framealpha=0.9, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved -> {save_path}")
    plt.show()


# Visestruke metrike u subplotovima 

def plot_multi_metric(
    df: pd.DataFrame,
    metrics: list[str] = None,
    resample: str = "5min",
    save_path: str = None,
):
    """
    Grid subplotova - svaki subplot prikazuje jednu metriku kroz vreme.
    Korisno za uporedjivanje vise indikatora istovremeno.
    """
    if metrics is None:
        metrics = ["byte_rate", "packet_rate", "avg_packet_size",
                   "tcp_syn_ratio", "udp_ratio", "src_ip_entropy"]

    n_cols = 2
    n_rows = (len(metrics) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        df_grouped = (
            df.set_index("ts_formated")
            .groupby([pd.Grouper(freq=resample), "label"])[metric]
            .mean()
            .reset_index()
        )

        for label, color in CLASS_COLORS.items():
            subset = df_grouped[df_grouped["label"] == label]
            if subset.empty:
                continue
            ax.scatter(
                subset["ts_formated"], subset[metric],
                color=color, s=4, alpha=0.6, label=label,
            )

        ax.set_title(metric, fontsize=11)
        ax.set_ylabel(_metric_label(metric), fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.3)


    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    # Zajednička legenda ispod grafika
    handles = [
        mpatches.Patch(color=color, label=label)
        for label, color in CLASS_COLORS.items()
        if label in df["label"].unique()
    ]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=5, fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.suptitle("Overview metrics", fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    plt.show()


#  Distribucija klasa (pie + bar) 

def plot_class_distribution(
    df: pd.DataFrame,
    save_path: str = None,
):
    """
    Levo: pie chart distribucije klasa.
    Desno: bar chart sa apsolutnim brojevima uzoraka.
    """
    counts = df["label"].value_counts()
    colors = [CLASS_COLORS.get(l, "#999999") for l in counts.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Pie chart
    ax1.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.82,
    )
    ax1.set_title("Class distribution (%)", fontsize=13)

    # Bar chart
    bars = ax2.barh(counts.index, counts.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax2.text(
            bar.get_width() + counts.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=9,
        )
    ax2.set_xlabel("Number of samples", fontsize=11)
    ax2.set_title("Apsolute nubmer of samples per class", fontsize=13)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved -> {save_path}")
    plt.show()


# Helper

def _metric_label(metric: str) -> str:
    labels = {
        "byte_rate":       "Byte rate (B/s)",
        "packet_rate":     "Packets/s",
        "avg_packet_size": "Avg packet size (B)",
        "tcp_syn_ratio":   "TCP SYN ratio",
        "udp_ratio":       "UDP ratio",
        "src_ip_entropy":  "Source IP entropy",
    }
    return labels.get(metric, metric)


# Sumarna funkcija za vizualizaciju

def visualize_dataset(csv_path: str, resample: str = "5min"):
    """
    Poziva sve 4 vizualizacije odjednom.
    """
    print(f"Loading dataset: {csv_path}")
    df = load_dataset(csv_path)
    print(f"Total samples: {len(df):,}  |  Classes: {sorted(df['label'].unique())}\n")

    plot_class_distribution(df,  save_path="dist_klasa.png")
    plot_traffic_timeline(df,    save_path="timeline_byte_rate.png", resample=resample)
    plot_multi_metric(df,        save_path="multi_metrika.png",      resample=resample)


if __name__ == "__main__":
    path = input("CSV file path: ").strip()
    # Kasnije doradi ovaj deo za interval resamplovanja
    print("Valid resample intervals: 1min, 5min, 10min, 15min, 30min, 1h, 2h, 6h")
    resample = input("Resample interval (default 5min): ").strip() or "5min"
    visualize_dataset(path, resample=resample)
