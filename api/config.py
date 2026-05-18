
import os
# Vidi da kasnije importujes ove sttvari iz torchn_nn.py i dataset_generator.py

# Putanje po potrebi promenit!
MODEL_PATH = os.getenv("MODEL_PATH", "ddos_lstm_attention.pt")
# MODEL_PATH = os.getenv("MODEL_PATH", "ddos_lstm.pt")

CLASS_LABELS = [
   'normal',
    'udp_flood_large',
    'dns_amplification', 
    'subnet_carpet_bombing', 
    'syn_flood',
    'icmp_flood', 
    'udp_flood_mixed',
    'ntp_amplification',
    'ack_flood'
]

NUM_CLASSES = len(CLASS_LABELS)

# Sliding window
# Mora biti isti kao window_size koriscen pri treniranju!
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "20"))


# Mora biti isti redosled kao pri generisanju trening podataka!
FEATURE_NAMES = [
    "packet_rate",
    "byte_rate",
    "avg_packet_size",
    "std_packet_size",
    "udp_ratio",
    "tcp_ratio",
    "icmp_ratio",
    "tcp_syn_ratio",
    "tcp_ack_ratio",
    "tcp_fin_ratio",
    "unique_src_ips",
    "unique_dst_ips",
    "unique_flows",
    "src_ip_entropy",
    "dst_ip_entropy",
    "dst_port_entropy",
    "top_dst_port_share",
    "top_src_ip_packet_share",
    "top_src_ip_byte_share",
    "dns_query_ratio",
    "dns_response_ratio",
    "dst_subnet_spread"
]


NUM_FEATURES = len(FEATURE_NAMES)

#Model arhitektura (mora odgovarati sacuvanom modelu)!
LSTM_HIDDEN_SIZE = int(os.getenv("LSTM_HIDDEN_SIZE", "128"))
LSTM_NUM_LAYERS = int(os.getenv("LSTM_NUM_LAYERS", "2"))
LSTM_DROPOUT = float(os.getenv("LSTM_DROPOUT", "0.3"))