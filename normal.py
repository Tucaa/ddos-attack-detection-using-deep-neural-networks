from functions import rand_uniform
import random


# Normal Web Traffic (HTTP/HTTPS)
def normal_web_traffic() -> dict:
    return {
        "label": "normal",
        # Volumetrija (normalan rate)
        "packet_rate": rand_uniform(2000, 12000),
        "byte_rate": rand_uniform(2000, 12000) * rand_uniform(400, 1200),
        "avg_packet_size": rand_uniform(400, 1200),
        "std_packet_size": rand_uniform(200, 500),
        # Protocol (TCP dominantan za web)
        "udp_ratio": rand_uniform(0.05, 0.20),
        "tcp_ratio": rand_uniform(0.75, 0.92),
        "icmp_ratio": rand_uniform(0.01, 0.05),
        # TCP flags (normalan handshake pattern)
        "tcp_syn_ratio": rand_uniform(0.08, 0.15),
        "tcp_ack_ratio": rand_uniform(0.70, 0.85),
        "tcp_fin_ratio": rand_uniform(0.05, 0.12),
        # IP diversity
        "unique_src_ips": rand_uniform(100, 800),
        "unique_dst_ips": rand_uniform(80, 600),
        "unique_flows": rand_uniform(200, 1200),
        "src_ip_entropy": rand_uniform(3.0, 5.0),
        "dst_ip_entropy": rand_uniform(2.5, 4.5),
        # Port patterns (HTTP/HTTPS - port 80, 443)
        "dst_port_entropy": rand_uniform(1.5, 3.0),
        "top_dst_port_share": rand_uniform(0.40, 0.70),
        # Top talkers (distribuirano)
        "top_src_ip_packet_share": rand_uniform(0.05, 0.20),
        "top_src_ip_byte_share": rand_uniform(0.05, 0.25),
        # DNS (normalna rezolucija)
        "dns_query_ratio": rand_uniform(0.02, 0.08),
        "dns_response_ratio": rand_uniform(0.02, 0.08),
        # Subnet spread
        "dst_subnet_spread": rand_uniform(0.15, 0.45),
    }


# Normal Business/Enterprise Traffic
def normal_enterprise_traffic() -> dict:
    return {
        "label": "normal",
        # Volumetrija
        "packet_rate": rand_uniform(3000, 15000),
        "byte_rate": rand_uniform(3000, 15000) * rand_uniform(300, 900),
        "avg_packet_size": rand_uniform(300, 900),
        "std_packet_size": rand_uniform(150, 400),
        # Protocol (mixed TCP/UDP za različite servise)
        "udp_ratio": rand_uniform(0.15, 0.35),
        "tcp_ratio": rand_uniform(0.60, 0.80),
        "icmp_ratio": rand_uniform(0.02, 0.08),
        # TCP flags
        "tcp_syn_ratio": rand_uniform(0.10, 0.18),
        "tcp_ack_ratio": rand_uniform(0.65, 0.80),
        "tcp_fin_ratio": rand_uniform(0.08, 0.15),
        # IP diversity
        "unique_src_ips": rand_uniform(200, 1200),
        "unique_dst_ips": rand_uniform(150, 900),
        "unique_flows": rand_uniform(400, 2000),
        "src_ip_entropy": rand_uniform(3.5, 5.5),
        "dst_ip_entropy": rand_uniform(3.0, 5.0),
        # Port patterns (raznovrsni business portovi)
        "dst_port_entropy": rand_uniform(2.5, 4.5),
        "top_dst_port_share": rand_uniform(0.20, 0.50),
        # Top talkers
        "top_src_ip_packet_share": rand_uniform(0.08, 0.25),
        "top_src_ip_byte_share": rand_uniform(0.10, 0.30),
        # DNS
        "dns_query_ratio": rand_uniform(0.03, 0.10),
        "dns_response_ratio": rand_uniform(0.03, 0.10),
        # Subnet spread
        "dst_subnet_spread": rand_uniform(0.20, 0.55),
    }


# Normal Streaming/Media Traffic
def normal_streaming_traffic() -> dict:
    return {
        "label": "normal",
        # Volumetrija (viši throughput, stabilniji)
        "packet_rate": rand_uniform(5000, 20000),
        "byte_rate": rand_uniform(5000, 20000) * rand_uniform(800, 1400),
        "avg_packet_size": rand_uniform(800, 1400),
        "std_packet_size": rand_uniform(100, 300),
        # Protocol (UDP za streaming, TCP za kontrolu)
        "udp_ratio": rand_uniform(0.60, 0.85),
        "tcp_ratio": rand_uniform(0.12, 0.35),
        "icmp_ratio": rand_uniform(0.01, 0.05),
        # TCP flags
        "tcp_syn_ratio": rand_uniform(0.05, 0.12),
        "tcp_ack_ratio": rand_uniform(0.70, 0.88),
        "tcp_fin_ratio": rand_uniform(0.03, 0.10),
        # IP diversity (limitiran broj streaming servera)
        "unique_src_ips": rand_uniform(80, 500),
        "unique_dst_ips": rand_uniform(20, 150),
        "unique_flows": rand_uniform(150, 800),
        "src_ip_entropy": rand_uniform(2.5, 4.5),
        "dst_ip_entropy": rand_uniform(1.5, 3.5),
        # Port patterns
        "dst_port_entropy": rand_uniform(1.0, 2.5),
        "top_dst_port_share": rand_uniform(0.50, 0.80),
        # Top talkers
        "top_src_ip_packet_share": rand_uniform(0.10, 0.35),
        "top_src_ip_byte_share": rand_uniform(0.15, 0.40),
        # DNS
        "dns_query_ratio": rand_uniform(0.01, 0.05),
        "dns_response_ratio": rand_uniform(0.01, 0.05),
        # Subnet spread
        "dst_subnet_spread": rand_uniform(0.10, 0.35),
    }


# Normal DNS Traffic
def normal_dns_traffic() -> dict:
    return {
        "label": "normal",
        # Volumetrija (nizak rate, mali paketi)
        "packet_rate": rand_uniform(1000, 6000),
        "byte_rate": rand_uniform(1000, 6000) * rand_uniform(80, 200),
        "avg_packet_size": rand_uniform(80, 200),
        "std_packet_size": rand_uniform(30, 80),
        # Protocol (UDP dominantan za DNS)
        "udp_ratio": rand_uniform(0.85, 0.95),
        "tcp_ratio": rand_uniform(0.03, 0.12),
        "icmp_ratio": rand_uniform(0.01, 0.05),
        # TCP flags
        "tcp_syn_ratio": rand_uniform(0.05, 0.15),
        "tcp_ack_ratio": rand_uniform(0.60, 0.80),
        "tcp_fin_ratio": rand_uniform(0.05, 0.15),
        # IP diversity
        "unique_src_ips": rand_uniform(150, 1000),
        "unique_dst_ips": rand_uniform(5, 30),
        "unique_flows": rand_uniform(200, 1500),
        "src_ip_entropy": rand_uniform(3.0, 5.5),
        "dst_ip_entropy": rand_uniform(1.0, 2.5),
        # Port patterns (port 53)
        "dst_port_entropy": rand_uniform(0.3, 1.0),
        "top_dst_port_share": rand_uniform(0.85, 0.98),
        # Top talkers
        "top_src_ip_packet_share": rand_uniform(0.03, 0.15),
        "top_src_ip_byte_share": rand_uniform(0.03, 0.15),
        # DNS (BALANCED query i response)
        "dns_query_ratio": rand_uniform(0.45, 0.55),
        "dns_response_ratio": rand_uniform(0.45, 0.55),
        # Subnet spread
        "dst_subnet_spread": rand_uniform(0.05, 0.20),
    }


# Normal Email/SMTP Traffic
def normal_email_traffic() -> dict:
    return {
        "label": "normal",
        # Volumetrija (nizak rate)
        "packet_rate": rand_uniform(800, 5000),
        "byte_rate": rand_uniform(800, 5000) * rand_uniform(300, 800),
        "avg_packet_size": rand_uniform(300, 800),
        "std_packet_size": rand_uniform(150, 400),
        # Protocol (TCP dominantan)
        "udp_ratio": rand_uniform(0.05, 0.15),
        "tcp_ratio": rand_uniform(0.80, 0.92),
        "icmp_ratio": rand_uniform(0.01, 0.05),
        # TCP flags
        "tcp_syn_ratio": rand_uniform(0.08, 0.15),
        "tcp_ack_ratio": rand_uniform(0.72, 0.87),
        "tcp_fin_ratio": rand_uniform(0.06, 0.13),
        # IP diversity
        "unique_src_ips": rand_uniform(100, 700),
        "unique_dst_ips": rand_uniform(50, 400),
        "unique_flows": rand_uniform(150, 900),
        "src_ip_entropy": rand_uniform(2.8, 4.8),
        "dst_ip_entropy": rand_uniform(2.3, 4.3),
        # Port patterns (SMTP ports: 25, 465, 587)
        "dst_port_entropy": rand_uniform(1.0, 2.5),
        "top_dst_port_share": rand_uniform(0.50, 0.80),
        # Top talkers
        "top_src_ip_packet_share": rand_uniform(0.08, 0.25),
        "top_src_ip_byte_share": rand_uniform(0.10, 0.30),
        # DNS
        "dns_query_ratio": rand_uniform(0.03, 0.10),
        "dns_response_ratio": rand_uniform(0.03, 0.10),
        # Subnet spread
        "dst_subnet_spread": rand_uniform(0.15, 0.45),
    }


# Izmesan Normal Traffic
def normal_mixed_traffic() -> dict:
    traffic_type = random.random()

    if traffic_type < 0.35:
        return normal_web_traffic()
    elif traffic_type < 0.60:
        return normal_enterprise_traffic()
    elif traffic_type < 0.80:
        return normal_streaming_traffic()
    elif traffic_type < 0.92:
        return normal_dns_traffic()
    else:
        return normal_email_traffic()