
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
    "packet_size",
    "packets_per_second",
    "bytes_per_second",
    "src_port",
    "dst_port",
    "protocol",
    "tcp_flags",
    "ttl",
    "inter_arrival_time",
    "flow_duration",
    "unique_src_ips",
    "unique_dst_ports",
]

NUM_FEATURES = len(FEATURE_NAMES)

#Model arhitektura (mora odgovarati sacuvanom modelu)!
LSTM_HIDDEN_SIZE = int(os.getenv("LSTM_HIDDEN_SIZE", "128"))
LSTM_NUM_LAYERS = int(os.getenv("LSTM_NUM_LAYERS", "2"))
LSTM_DROPOUT = float(os.getenv("LSTM_DROPOUT", "0.3"))