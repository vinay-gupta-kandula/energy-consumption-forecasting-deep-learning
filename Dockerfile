FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

# install CPU torch FIRST (prevents lightning reinstalling it)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    torchaudio==2.2.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# now install remaining deps
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MPLBACKEND=Agg
ENV WANDB_MODE=disabled
ENV PYTHONUNBUFFERED=1

CMD ["python","-m","src.pipeline"]

