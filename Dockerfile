FROM python:3.11-slim

WORKDIR /app

# Sistemske zavisnosti (curl za healthcheck, gcc za neke Python pakete)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# PyTorch mora odvojeno — +cu121 verzije nisu na standardnom PyPI
RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Za tensoflow i cudu
RUN pip install --no-cache-dir tensorflow[and-cuda]==2.21.0

# Ostale zavisnosti
RUN pip install --no-cache-dir -r requirements.txt

# COPY . . je uklonjen — kod dolazi preko volume monta u docker-compose.yml
# Na ovaj nacin image ne sadrzi source code, a --reload radi ispravno

EXPOSE 8000