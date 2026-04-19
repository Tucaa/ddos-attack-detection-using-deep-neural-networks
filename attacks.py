from functions import rand_uniform, rand_normal, clamp


# Abnormalna velicina UDP paketa
def udp_large_packets() -> dict:
    packet_rate = rand_normal(80000, 15000)
    byte_rate = packet_rate * rand_normal(1300, 100)
    unique_src = rand_uniform(500, 3000)

    return {
        "label": "udp-flood-large",

        # Volumetrija
        "packet_rate": clamp(packet_rate, 40000, 150000),
        "byte_rate": clamp(byte_rate, 50000000, 200000000),
        "avg_packet_size": rand_normal(1300, 100),
        "std_packet_size": rand_uniform(50, 150),
        # Protocol distribucija
        "udp_ratio": rand_uniform(0.90, 0.99),
        "tcp_ratio": rand_uniform(0.01, 0.08),
        "icmp_ratio": rand_uniform(0.0, 0.02),
        # TCP flags (nizak jer je UDP dominantan)
        "tcp_syn_ratio": rand_uniform(0.0, 0.05),
        "tcp_ack_ratio": rand_uniform(0.0, 0.05),
        "tcp_fin_ratio": rand_uniform(0.0, 0.02),
        # IP
        "unique_src_ips": unique_src,
        "unique_dst_ips": rand_uniform(5, 50),
        "unique_flows": clamp(unique_src * rand_uniform(1.2, 2.0), 600, 5000),
        "src_ip_entropy": rand_uniform(4.0, 6.0),
        "dst_ip_entropy": rand_uniform(1.0, 2.5),
        # Patterni portova
        "dst_port_entropy": rand_uniform(0.5, 2.0),
        "top_dst_port_share": rand_uniform(0.30, 0.70),
        # Top talkeri
        "top_src_ip_packet_share": rand_uniform(0.03, 0.15),
        "top_src_ip_byte_share": rand_uniform(0.03, 0.15),
        # DNS (nizak jer nije DNS napad)
        "dns_query_ratio": rand_uniform(0.0, 0.02),
        "dns_response_ratio": rand_uniform(0.0, 0.02),
        # Subnet spread
        "dst_subnet_spread": rand_uniform(0.05, 0.25),
    }


# DNS Amplification
def dns_amplification() -> dict:
    packet_rate = rand_normal(60000, 12000)
    unique_src = rand_uniform(2000, 10000)

    return {
        "label": "dns-amplification",

        # Volumetrija (veliki paketi zbog DNS responsa)
        "packet_rate": clamp(packet_rate, 30000, 120000),
        "byte_rate": packet_rate * rand_uniform(800, 1200),
        "avg_packet_size": rand_uniform(900, 1400),
        "std_packet_size": rand_uniform(100, 300),
        # Protocol (UDP dominantan)
        "udp_ratio": rand_uniform(0.92, 0.99),
        "tcp_ratio": rand_uniform(0.01, 0.05),
        "icmp_ratio": rand_uniform(0.0, 0.03),
        # TCP flags
        "tcp_syn_ratio": rand_uniform(0.0, 0.03),
        "tcp_ack_ratio": rand_uniform(0.0, 0.03),
        "tcp_fin_ratio": rand_uniform(0.0, 0.01),
        # IP diversity (mnogo source IP)
        "unique_src_ips": unique_src,
        "unique_dst_ips": rand_uniform(5, 100),
        "unique_flows": clamp(unique_src * rand_uniform(0.8, 1.5), 2000, 15000),
        "src_ip_entropy": rand_uniform(5.0, 7.0),
        "dst_ip_entropy": rand_uniform(1.5, 3.0),
        # Port patterns (port 53 dominantan)
        "dst_port_entropy": rand_uniform(0.3, 1.0),
        "top_dst_port_share": rand_uniform(0.80, 0.98),
        # Top talkers (distribuirano)
        "top_src_ip_packet_share": rand_uniform(0.01, 0.08),
        "top_src_ip_byte_share": rand_uniform(0.01, 0.08),
        # DNS specific (KLJUČNA karakteristika!)
        "dns_query_ratio": rand_uniform(0.02, 0.10),
        "dns_response_ratio": rand_uniform(0.85, 0.98),
        # Subnet spread
        "dst_subnet_spread": rand_uniform(0.10, 0.40),
    }


# Subnet carpet Bombing
def subnet_carpet_bombing() -> dict:
    packet_rate = rand_normal(70000, 18000)
    unique_dst = rand_uniform(5000, 20000)

    return {
        "label": "subnet-carpet-bombing",
        # Volumetrija
        "packet_rate": clamp(packet_rate, 35000, 140000),
        "byte_rate": packet_rate * rand_uniform(100, 400),
        "avg_packet_size": rand_uniform(64, 250),
        "std_packet_size": rand_uniform(20, 80),
        # Protocol (može biti mixed)
        "udp_ratio": rand_uniform(0.50, 0.85),
        "tcp_ratio": rand_uniform(0.10, 0.45),
        "icmp_ratio": rand_uniform(0.02, 0.15),
        # TCP flags
        "tcp_syn_ratio": rand_uniform(0.05, 0.30),
        "tcp_ack_ratio": rand_uniform(0.05, 0.25),
        "tcp_fin_ratio": rand_uniform(0.01, 0.10),
        # IP diversity (KLJUČNO: mnogo destination IPs u subnet-u)
        "unique_src_ips": rand_uniform(1000, 8000),
        "unique_dst_ips": unique_dst,
        "unique_flows": clamp(unique_dst * rand_uniform(0.9, 1.3), 5000, 25000),
        "src_ip_entropy": rand_uniform(4.5, 6.5),
        "dst_ip_entropy": rand_uniform(6.0, 8.0),  # Visoka dst entropija!
        # Port patterns
        "dst_port_entropy": rand_uniform(1.5, 4.0),
        "top_dst_port_share": rand_uniform(0.10, 0.35),
        # Top talkers
        "top_src_ip_packet_share": rand_uniform(0.01, 0.06),
        "top_src_ip_byte_share": rand_uniform(0.01, 0.06),
        # DNS
        "dns_query_ratio": rand_uniform(0.0, 0.05),
        "dns_response_ratio": rand_uniform(0.0, 0.05),
        # Subnet spread (KLJUČNA karakteristika - visoka!)
        "dst_subnet_spread": rand_uniform(0.70, 0.95),
    }


# SYN Flood Attack
def syn_flood() -> dict:
    packet_rate = rand_normal(90000, 20000)
    unique_src = rand_uniform(3000, 15000)

    return {
        "label": "syn-flood",
        # Volumetrija (manji paketi - samo SYN)
        "packet_rate": clamp(packet_rate, 50000, 180000),
        "byte_rate": packet_rate * rand_uniform(60, 100),
        "avg_packet_size": rand_uniform(54, 80),  # SYN paketi su mali
        "std_packet_size": rand_uniform(5, 20),
        # Protocol (TCP dominantan)
        "udp_ratio": rand_uniform(0.0, 0.05),
        "tcp_ratio": rand_uniform(0.92, 0.99),
        "icmp_ratio": rand_uniform(0.0, 0.03),
        # TCP flags (KLJUČNO: visok SYN, nizak ACK)
        "tcp_syn_ratio": rand_uniform(0.85, 0.98),  # Skoro sve SYN
        "tcp_ack_ratio": rand_uniform(0.02, 0.10),  # Malo ACK
        "tcp_fin_ratio": rand_uniform(0.0, 0.03),
        # IP diversity (mnogo source IPs, mali broj dest IPs)
        "unique_src_ips": unique_src,
        "unique_dst_ips": rand_uniform(5, 100),
        "unique_flows": clamp(unique_src * rand_uniform(0.9, 1.5), 3000, 20000),
        "src_ip_entropy": rand_uniform(5.0, 7.0),
        "dst_ip_entropy": rand_uniform(1.5, 3.0),
        # Port patterns (često targetuje specifične portove - 80, 443)
        "dst_port_entropy": rand_uniform(0.5, 2.0),
        "top_dst_port_share": rand_uniform(0.40, 0.80),
        # Top talkers
        "top_src_ip_packet_share": rand_uniform(0.02, 0.10),
        "top_src_ip_byte_share": rand_uniform(0.02, 0.10),
        # DNS
        "dns_query_ratio": rand_uniform(0.0, 0.02),
        "dns_response_ratio": rand_uniform(0.0, 0.02),
        # Subnet spread
        "dst_subnet_spread": rand_uniform(0.05, 0.30),
    }


# ICMP Flood Attack
def icmp_flood() -> dict:
    packet_rate = rand_normal(100000, 25000)
    unique_src = rand_uniform(2000, 12000)

    return {
        "label": "icmp-flood",

        # Volumetrija (mali paketi ICMP echo request)
        "packet_rate": clamp(packet_rate, 60000, 200000),
        "byte_rate": packet_rate * rand_uniform(70, 120),
        "avg_packet_size": rand_uniform(64, 128),
        "std_packet_size": rand_uniform(10, 40),
        # Protocol (ICMP dominantan)
        "udp_ratio": rand_uniform(0.0, 0.05),
        "tcp_ratio": rand_uniform(0.0, 0.05),
        "icmp_ratio": rand_uniform(0.90, 0.99),
        # TCP flags (zanemarljivo jer nema TCP)
        "tcp_syn_ratio": rand_uniform(0.0, 0.02),
        "tcp_ack_ratio": rand_uniform(0.0, 0.02),
        "tcp_fin_ratio": rand_uniform(0.0, 0.01),
        # IP diversity
        "unique_src_ips": unique_src,
        "unique_dst_ips": rand_uniform(10, 150),
        "unique_flows": clamp(unique_src * rand_uniform(0.8, 1.3), 2000, 15000),
        "src_ip_entropy": rand_uniform(4.5, 6.5),
        "dst_ip_entropy": rand_uniform(2.0, 3.5),
        # Port patterns (ICMP nema portove ali možemo koristiti type/code)
        "dst_port_entropy": rand_uniform(0.0, 0.5),
        "top_dst_port_share": rand_uniform(0.80, 0.98),
        # Top talkers
        "top_src_ip_packet_share": rand_uniform(0.02, 0.12),
        "top_src_ip_byte_share": rand_uniform(0.02, 0.12),
        # DNS
        "dns_query_ratio": 0.0,
        "dns_response_ratio": 0.0,
        # Subnet spread
        "dst_subnet_spread": rand_uniform(0.10, 0.40),
    }


# UDP Flood (Generic/Mixed ports)
def udp_flood_mixed() -> dict:
    packet_rate = rand_normal(85000, 18000)
    unique_src = rand_uniform(1500, 8000)

    return {
        "label": "udp-flood-mixed",
        # Volumetrija (srednje veličine paketi)
        "packet_rate": clamp(packet_rate, 45000, 160000),
        "byte_rate": packet_rate * rand_uniform(200, 600),
        "avg_packet_size": rand_uniform(150, 500),
        "std_packet_size": rand_uniform(50, 150),
        # Protocol (UDP dominantan)
        "udp_ratio": rand_uniform(0.88, 0.98),
        "tcp_ratio": rand_uniform(0.01, 0.10),
        "icmp_ratio": rand_uniform(0.01, 0.05),
        # TCP flags
        "tcp_syn_ratio": rand_uniform(0.02, 0.08),
        "tcp_ack_ratio": rand_uniform(0.02, 0.08),
        "tcp_fin_ratio": rand_uniform(0.0, 0.03),
        # IP diversity
        "unique_src_ips": unique_src,
        "unique_dst_ips": rand_uniform(10, 200),
        "unique_flows": clamp(unique_src * rand_uniform(1.0, 2.0), 2000, 12000),
        "src_ip_entropy": rand_uniform(4.0, 6.0),
        "dst_ip_entropy": rand_uniform(2.0, 4.0),
        # Port patterns (KLJUČNO: mixed ports, viša entropija)
        "dst_port_entropy": rand_uniform(2.5, 4.5),
        "top_dst_port_share": rand_uniform(0.15, 0.40),
        # Top talkers
        "top_src_ip_packet_share": rand_uniform(0.03, 0.15),
        "top_src_ip_byte_share": rand_uniform(0.03, 0.15),
        # DNS
        "dns_query_ratio": rand_uniform(0.0, 0.05),
        "dns_response_ratio": rand_uniform(0.0, 0.05),
        # Subnet spread
        "dst_subnet_spread": rand_uniform(0.10, 0.35),
    }


# NTP Amplification Attack port 123!
def ntp_amplification() -> dict:
    packet_rate = rand_normal(55000, 12000)
    unique_src = rand_uniform(1500, 7000)

    return {
        "label": "ntp-amplification",
        # Volumetrija (veliki response paketi)
        "packet_rate": clamp(packet_rate, 30000, 110000),
        "byte_rate": packet_rate * rand_uniform(400, 700),
        "avg_packet_size": rand_uniform(400, 600),
        "std_packet_size": rand_uniform(50, 120),
        # Protocol (UDP - port 123)
        "udp_ratio": rand_uniform(0.90, 0.99),
        "tcp_ratio": rand_uniform(0.01, 0.07),
        "icmp_ratio": rand_uniform(0.0, 0.03),
        # TCP flags
        "tcp_syn_ratio": rand_uniform(0.0, 0.05),
        "tcp_ack_ratio": rand_uniform(0.0, 0.05),
        "tcp_fin_ratio": rand_uniform(0.0, 0.02),
        # IP diversity (mnogo reflektora)
        "unique_src_ips": unique_src,
        "unique_dst_ips": rand_uniform(5, 80),
        "unique_flows": clamp(unique_src * rand_uniform(0.8, 1.4), 1500, 9000),
        "src_ip_entropy": rand_uniform(4.5, 6.5),
        "dst_ip_entropy": rand_uniform(1.5, 3.0),
        # Port patterns (port 123 dominantan)
        "dst_port_entropy": rand_uniform(0.3, 1.2),
        "top_dst_port_share": rand_uniform(0.75, 0.95),
        # Top talkers
        "top_src_ip_packet_share": rand_uniform(0.02, 0.10),
        "top_src_ip_byte_share": rand_uniform(0.02, 0.10),
        # DNS (nije DNS attack)
        "dns_query_ratio": rand_uniform(0.0, 0.02),
        "dns_response_ratio": rand_uniform(0.0, 0.02),
        # Subnet spread
        "dst_subnet_spread": rand_uniform(0.08, 0.35),
    }


# ACK Flood Attack
def ack_flood() -> dict:
    packet_rate = rand_normal(95000, 22000)
    unique_src = rand_uniform(4000, 18000)

    return {
        "label": "ack-flood",
        # Volumetrija
        "packet_rate": clamp(packet_rate, 50000, 190000),
        "byte_rate": packet_rate * rand_uniform(50, 90),
        "avg_packet_size": rand_uniform(50, 100),
        "std_packet_size": rand_uniform(8, 25),
        # Protocol (TCP dominantan)
        "udp_ratio": rand_uniform(0.0, 0.05),
        "tcp_ratio": rand_uniform(0.92, 0.99),
        "icmp_ratio": rand_uniform(0.0, 0.03),
        # TCP flags (KLJUČNO: visok ACK nizak SYN)
        "tcp_syn_ratio": rand_uniform(0.01, 0.08),
        "tcp_ack_ratio": rand_uniform(0.88, 0.98),  # Skoro sve ACK
        "tcp_fin_ratio": rand_uniform(0.01, 0.05),
        # IP diversity
        "unique_src_ips": unique_src,
        "unique_dst_ips": rand_uniform(10, 150),
        "unique_flows": clamp(unique_src * rand_uniform(0.8, 1.4), 4000, 22000),
        "src_ip_entropy": rand_uniform(5.0, 7.0),
        "dst_ip_entropy": rand_uniform(2.0, 3.5),
        # Port patterns
        "dst_port_entropy": rand_uniform(0.8, 2.5),
        "top_dst_port_share": rand_uniform(0.35, 0.70),
        # Top talkers
        "top_src_ip_packet_share": rand_uniform(0.02, 0.12),
        "top_src_ip_byte_share": rand_uniform(0.02, 0.12),
        # DNS
        "dns_query_ratio": rand_uniform(0.0, 0.02),
        "dns_response_ratio": rand_uniform(0.0, 0.02),
        # Subnet spread
        "dst_subnet_spread": rand_uniform(0.10, 0.40),
    }


# F-ja za generisanje podataka
# Izmesaj i ovde podatke
ATTACK_GENERATORS = {
    "udp-flood-large": udp_large_packets,
    "dns-amplification": dns_amplification,
    "subnet-carpet-bombing": subnet_carpet_bombing,
    "syn-flood": syn_flood,
    "icmp-flood": icmp_flood,
    "udp-flood-mixed": udp_flood_mixed,
    "ntp-amplification": ntp_amplification,
    "ack-flood": ack_flood,
}
