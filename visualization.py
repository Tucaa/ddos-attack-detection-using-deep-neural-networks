import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import gc  # za manuelno oslobadjanje memorije

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

# Kolone koje su nam potrebne - ucitavamo samo ove, ne sve
NEEDED_COLS = [
    "ts_formated", "label",
    "byte_rate", "packet_rate", "avg_packet_size",
    "tcp_syn_ratio", "udp_ratio", "src_ip_entropy",
]


def load_dataset(csv_path: str, resample: str = "5min") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ucitava CSV u chunkovima i odmah agregira podatke.
    Vraca dva DataFrame-a:
      - df_resampled: agregirani podaci po (vreme, label) za vizualizacije
      - df_counts: broj uzoraka po klasi za distribuciju
    
    Na ovaj nacin nikad u RAM-u nema vise od jednog chunka sirovih podataka.
    """
    CHUNK_SIZE = 200_000
    
    agg_chunks = []
    count_chunks = []

    print("Reading CSV in chunks...")
    for i, chunk in enumerate(pd.read_csv(
        csv_path,
        usecols=NEEDED_COLS,          # samo kolone koje koristimo
        parse_dates=["ts_formated"],
        chunksize=CHUNK_SIZE,
    )):
        # label kao kategoricki tip - stedi memoriju
        chunk["label"] = chunk["label"].astype("category")
        chunk = chunk.sort_values("ts_formated")

        # Agregacija unutar chunka pre nego sto ga cuvamo
        chunk_agg = (
            chunk.set_index("ts_formated")
            .groupby([pd.Grouper(freq=resample), "label"])
            .agg(
                byte_rate=("byte_rate", "mean"),
                packet_rate=("packet_rate", "mean"),
                avg_packet_size=("avg_packet_size", "mean"),
                tcp_syn_ratio=("tcp_syn_ratio", "mean"),
                udp_ratio=("udp_ratio", "mean"),
                src_ip_entropy=("src_ip_entropy", "mean"),
            )
            .reset_index()
        )
        agg_chunks.append(chunk_agg)
        
        # Brojevi klasa po chunku
        count_chunks.append(chunk["label"].value_counts())

        del chunk  # eksplicitno brisanje chunka iz memorije
        if i % 5 == 0:
            gc.collect()
            print(f"  Processed chunk {i+1}...")

    print("Merging aggregated chunks...")
    
    # Spajanje i ponovna agregacija (agregirani chunkovi su mali)
    df_agg = pd.concat(agg_chunks, ignore_index=True)
    del agg_chunks
    gc.collect()

    df_resampled = (
        df_agg.set_index("ts_formated")
        .groupby([pd.Grouper(freq=resample), "label"])
        .mean()
        .reset_index()
    )
    del df_agg
    gc.collect()

    # Finalni brojaci klasa
    df_counts = pd.concat(count_chunks).groupby(level=0).sum().sort_values(ascending=False)
    
    print(f"Done. Aggregated shape: {df_resampled.shape}")
    return df_resampled, df_counts


def plot_traffic_timeline(
    df_resampled: pd.DataFrame,
    metric: str = "byte_rate",
    title: str = "Network traffic through time",
    save_path: str = None,
):
    """Prima vec agregirani DataFrame umesto sirovih podataka."""
    fig, ax = plt.subplots(figsize=(18, 6))

    for label, color in CLASS_COLORS.items():
        subset = df_resampled[df_resampled["label"] == label]
        if subset.empty:
            continue
        ax.scatter(
            subset["ts_formated"], subset[metric],
            color=color, s=6, alpha=0.7, label=label, zorder=2,
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
    plt.close()  # oslobadjanje figure iz memorije


def plot_multi_metric(
    df_resampled: pd.DataFrame,
    metrics: list[str] = None,
    save_path: str = None,
):
    """Prima vec agregirani DataFrame umesto sirovih podataka."""
    if metrics is None:
        metrics = ["byte_rate", "packet_rate", "avg_packet_size",
                   "tcp_syn_ratio", "udp_ratio", "src_ip_entropy"]

    n_cols = 2
    n_rows = (len(metrics) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        for label, color in CLASS_COLORS.items():
            subset = df_resampled[df_resampled["label"] == label]
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

    handles = [
        mpatches.Patch(color=color, label=label)
        for label, color in CLASS_COLORS.items()
        if label in df_resampled["label"].unique()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle("Overview metrics", fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    plt.show()
    plt.close()


def plot_class_distribution(
    df_counts: pd.Series,
    save_path: str = None,
):
    """
    Prima Series sa brojevima klasa umesto celog DataFrame-a.
    """
    colors = [CLASS_COLORS.get(l, "#999999") for l in df_counts.index]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.pie(df_counts.values, labels=df_counts.index, colors=colors,
            autopct="%1.1f%%", startangle=140, pctdistance=0.82)
    ax1.set_title("Class distribution (%)", fontsize=13)

    bars = ax2.barh(df_counts.index, df_counts.values,
                    color=colors, edgecolor="white")
    for bar, val in zip(bars, df_counts.values):
        ax2.text(
            bar.get_width() + df_counts.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=9,
        )
    ax2.set_xlabel("Number of samples", fontsize=11)
    ax2.set_title("Absolute number of samples per class", fontsize=13)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved -> {save_path}")
    plt.show()
    plt.close()


def plot_byte_timeline(
    df_resampled: pd.DataFrame,
    save_path: str = None,
):
    """
    Popravljen plot_byte_timeline - bez iterrows().
    Koristi vektorizovano grupiranje uzastopnih perioda iste klase
    umesto petlje po svakom redu.
    """
    fig, ax = plt.subplots(figsize=(20, 6))

    # Ukupni byte_rate po vremenu (suma svih klasa u tom intervalu)
    df_total = (
        df_resampled.groupby("ts_formated")["byte_rate"]
        .sum()
        .reset_index()
        .sort_values("ts_formated")
    )
    ax.plot(df_total["ts_formated"], df_total["byte_rate"],
            color="#2c3e50", linewidth=0.9, alpha=0.85, zorder=3,
            label="Total byte rate")

    # Dominantna klasa po vremenskom intervalu (umesto iterrows po svakom redu)
    df_dominant = (
        df_resampled.loc[
            df_resampled.groupby("ts_formated")["byte_rate"].idxmax()
        ]
        [["ts_formated", "label"]]
        .sort_values("ts_formated")
        .reset_index(drop=True)
    )

    # Grupisanje uzastopnih intervala iste klase u jedan span
    # Umesto miliona axvspan poziva - samo onoliko koliko ima promena klase
    df_dominant["group"] = (
        df_dominant["label"] != df_dominant["label"].shift()
    ).cumsum()

    for _, grp in df_dominant.groupby("group"):
        lbl   = grp["label"].iloc[0]
        t_start = grp["ts_formated"].iloc[0]
        t_end   = grp["ts_formated"].iloc[-1]
        color = CLASS_COLORS.get(lbl, "#999999")
        alpha = 0.10 if lbl == "normal" else 0.25
        ax.axvspan(t_start, t_end, color=color, alpha=alpha, linewidth=0)

    handles = [
        mpatches.Patch(color=CLASS_COLORS.get(lbl, "#999999"), label=lbl, alpha=0.6)
        for lbl in sorted(df_resampled["label"].unique())
    ]
    handles.insert(0, plt.Line2D([0], [0], color="#2c3e50",
                                  linewidth=1.5, label="Byte rate"))

    ax.set_title("Byte rate kroz vreme po tipu saobracaja", fontsize=14, pad=14)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Byte rate (B/s)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.xticks(rotation=35)
    ax.legend(handles=handles, loc="upper left", fontsize=8,
              framealpha=0.9, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved -> {save_path}")
    plt.show()
    plt.close()


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


def visualize_dataset(csv_path: str, resample: str = "5min"):
    """
    Ucitava dataset jednom, agregira tokom citanja,
    pa sve 4 vizualizacije rade nad malim agregiranim DataFrame-om.
    """
    print(f"Loading dataset: {csv_path}")
    df_resampled, df_counts = load_dataset(csv_path, resample=resample)
    
    labels = sorted(df_resampled["label"].unique())
    total  = df_counts.sum()
    print(f"Total samples: {total:,}  |  Classes: {labels}\n")

    plot_byte_timeline(df_resampled,  save_path="byte_timeline.png")
    plot_class_distribution(df_counts, save_path="dist_klasa.png")
    plot_traffic_timeline(df_resampled, save_path="timeline_byte_rate.png")
    plot_multi_metric(df_resampled,   save_path="multi_metrika.png")


if __name__ == "__main__":
    path = input("CSV file path: ").strip()
    print("Valid resample intervals: 1min, 5min, 10min, 15min, 30min, 1h, 2h, 6h")
    resample = input("Resample interval (default 5min): ").strip() or "5min"
    visualize_dataset(path, resample=resample)