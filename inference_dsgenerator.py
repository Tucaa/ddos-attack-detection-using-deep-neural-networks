"""
Rekurzivni procesor za CIC-DDoS2019 dataset koji mapira podatke na specifičan skup od 22 feature-a.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path


# Ciljna lista feature-a (Redosled je ključan za model!)
TARGET_FEATURES = [
    'packet_rate', 'byte_rate', 'avg_packet_size', 'std_packet_size',
    'udp_ratio', 'tcp_ratio', 'icmp_ratio', 'tcp_syn_ratio',
    'tcp_ack_ratio', 'tcp_fin_ratio', 'unique_src_ips', 'unique_dst_ips',
    'unique_flows', 'src_ip_entropy', 'dst_ip_entropy', 'dst_port_entropy',
    'top_dst_port_share', 'top_src_ip_packet_share', 'top_src_ip_byte_share',
    'dns_query_ratio', 'dns_response_ratio', 'dst_subnet_spread'
]

# Mapiranje originalnih CIC labela na klase modela
LABEL_MAP = {
    "BENIGN": "normal", "Syn": "syn_flood", "SYN": "syn_flood",
    "UDP": "udp_flood_mixed", "UDPLag": "udp_flood_mixed",
    "LDAP": "dns_amplification", "MSSQL": "ntp_amplification",
    "NetBIOS": "udp_flood_large", "TFTP": "udp_flood_large",
    "DNS": "dns_amplification", "NTP": "ntp_amplification",
    "ICMP": "icmp_flood", "ACK": "ack_flood",
}

# Alijasi za kolone jer CICFlowMeter često menja nazive (razmaci, velika slova)
CIC_COL_ALIASES = {
    "protocol":        [" Protocol", "Protocol"],
    "flow_pkts_s":     [" Flow Packets/s", "Flow Packets/s", "flow_pkts_s"],
    "flow_bytes_s":    [" Flow Bytes/s", "Flow Bytes/s", "flow_bytes_s"],
    "avg_pkt_size":    [" Average Packet Size", "Avg Packet Size", "Packet Length Mean"],
    "std_pkt_size":    [" Packet Length Std", "Packet Length Std"],
    "total_pkts":      [" Total Fwd Packets", "Total Backward Packets"], # Koristi se za procente
    "syn_flag":        [" SYN Flag Count", "SYN Flag Count"],
    "ack_flag":        [" ACK Flag Count", "ACK Flag Count"],
    "fin_flag":        [" FIN Flag Count", "FIN Flag Count"],
    "label":           [" Label", "Label", "label"],
    "src_ip":          [" Source IP", "Src IP", "source_ip"],
    "dst_ip":          [" Destination IP", "Dst IP", "destination_ip"],
    "dst_port":        [" Destination Port", "Dst Port", "destination_port"],
}

def resolve_col(df: pd.DataFrame, canonical: str) -> str | None:
    """Pronalazi naziv kolone u DataFrame-u na osnovu liste poznatih alijasa."""
    for alias in CIC_COL_ALIASES.get(canonical, []):
        if alias in df.columns:
            return alias
    return None

def get_col(df: pd.DataFrame, canonical: str, default=0.0) -> pd.Series:
    """Izvlači kolonu, pretvara u brojeve i popunjava praznine default vrednošću."""
    col = resolve_col(df, canonical)
    if col:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)

def compute_unique_rolling(df: pd.DataFrame, window: int = 100):
    """Aproksimacija broja jedinstvenih IP adresa koristeći rolling prozor."""
    src_ip_col = resolve_col(df, "src_ip")
    dst_ip_col = resolve_col(df, "dst_ip")
    
    if src_ip_col:
        unique_src = df[src_ip_col].astype('category').cat.codes.rolling(window, min_periods=1).apply(lambda x: len(set(x)), raw=True)
    else:
        unique_src = pd.Series(1.0, index=df.index)
        
    if dst_ip_col:
        unique_dst = df[dst_ip_col].astype('category').cat.codes.rolling(window, min_periods=1).apply(lambda x: len(set(x)), raw=True)
    else:
        unique_dst = pd.Series(1.0, index=df.index)
        
    return unique_src, unique_dst

def preprocess(df: pd.DataFrame, rolling_window: int = 100) -> tuple[pd.DataFrame, pd.Series | None]:
    """Glavna funkcija koja transformiše CIC podatke u traženi skup od 22 feature-a."""
    
    # 1. Obrada labela
    label_col = resolve_col(df, "label")
    labels = df[label_col].str.strip().map(LABEL_MAP).fillna("unknown") if label_col else None
    
    # 2. Mapiranje osnovnih metričkih podataka
    out = pd.DataFrame(index=df.index)
    out['packet_rate']    = get_col(df, "flow_pkts_s")
    out['byte_rate']      = get_col(df, "flow_bytes_s")
    out['avg_packet_size'] = get_col(df, "avg_pkt_size")
    out['std_packet_size'] = get_col(df, "std_pkt_size")
    
    # 3. Odnosi protokola (Pretvaranje ID-a u binarni odnos)
    proto = get_col(df, "protocol")
    out['tcp_ratio']  = (proto == 6).astype(float)
    out['udp_ratio']  = (proto == 17).astype(float)
    out['icmp_ratio'] = (proto == 1).astype(float)
    
    # 4. Odnosi TCP zastavica (Aproksimacija na nivou flow-a)
    total_pkts = get_col(df, "total_pkts").replace(0, 1)
    out['tcp_syn_ratio'] = get_col(df, "syn_flag") / total_pkts
    out['tcp_ack_ratio'] = get_col(df, "ack_flag") / total_pkts
    out['tcp_fin_ratio'] = get_col(df, "fin_flag") / total_pkts
    
    # 5. Statistika jedinstvenih entiteta
    u_src, u_dst = compute_unique_rolling(df, rolling_window)
    out['unique_src_ips'] = u_src
    out['unique_dst_ips'] = u_dst
    out['unique_flows']   = pd.Series(1.0, index=df.index).rolling(rolling_window, min_periods=1).sum()
    
    # 6. Postavljanje nedostajućih feature-a na 0.0 (Entropije i kompleksni DNS podaci)
    # Ovi podaci se ne nalaze direktno u CIC CSV fajlovima i moraju biti inicijalizovani za model
    missing_cols = [
        'src_ip_entropy', 'dst_ip_entropy', 'dst_port_entropy',
        'top_dst_port_share', 'top_src_ip_packet_share', 'top_src_ip_byte_share',
        'dns_query_ratio', 'dns_response_ratio', 'dst_subnet_spread'
    ]
    for col in missing_cols:
        out[col] = 0.0

    # Osiguravanje ispravnog redosleda kolona
    out = out[TARGET_FEATURES]
    
    # Čišćenje: zamena beskonačnih vrednosti i NaN nulu
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = out.clip(lower=0.0)
    
    return out, labels

def main():
    """Ulazna tačka programa: upravljanje korisničkim unosom i iteracija kroz fajlove."""
    print("--- Recursive Preprocessor: Mapping to Custom Feature Set ---")
    
    folder_path = input("Enter path to input folder: ").strip()
    output_file = input("Enter path for output features CSV: ").strip()
    labels_file = input("Enter path for output labels CSV (Enter to skip): ").strip()
    
    sample_per_file = input("Rows per file (Enter for ALL): ").strip()
    sample_per_file = int(sample_per_file) if sample_per_file else None
    
    window = input("Rolling window size (Enter for 100): ").strip()
    window = int(window) if window else 100

    base_dir = Path(folder_path)
    if not base_dir.is_dir():
        print("Invalid directory.")
        return

    # Rekurzivno pronalaženje svih CSV fajlova
    csv_files = list(base_dir.rglob("*.csv"))
    all_features, all_labels = [], []

    for i, file_path in enumerate(csv_files):
        print(f"[{i+1}/{len(csv_files)}] Processing {file_path.name}...")
        try:
            # Čitanje ograničenog broja redova iz svakog fajla
            df = pd.read_csv(file_path, nrows=sample_per_file, low_memory=False)
            if df.empty: continue
            
            features, labels = preprocess(df, rolling_window=window)
            all_features.append(features)
            if labels is not None: all_labels.append(labels)
        except Exception as e:
            print(f"  Error: {e}")

    # Spajanje rezultata i čuvanje u finalne fajlove
    if all_features:
        final_features = pd.concat(all_features, ignore_index=True)
        final_features.to_csv(output_file, index=False)
        print(f"\nSaved {len(final_features):,} rows to {output_file}")
        
        if labels_file and all_labels:
            final_labels = pd.concat(all_labels, ignore_index=True)
            final_labels.to_csv(labels_file, index=False, header=["label"])
            print(f"Labels saved to {labels_file}")
    
    print("\nDone.")

if __name__ == "__main__":
    main()