# SeatSense on an NVIDIA GPU host. Build from the repository root:
#   docker build -f deploy/docker/Dockerfile -t seatsense .
# Code lives in the image; runtime data (models/, recordings/, states/, logs/, certs/, config/zones/,
# users.json, .env) is mounted at /data -- see compose.yaml.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# torch / torchvision come with the base image; the rest as tested (requirements.txt)
RUN pip install --no-cache-dir ultralytics==8.4.51 opencv-python-headless==4.13.0.92 numpy==2.4.6 pillow==11.3.0

COPY seatsense seatsense
COPY training training
COPY labelling labelling
COPY scripts scripts

ENV SEATSENSE_HOME=/data \
    SEATSENSE_BIND=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp
# runs as the data directory's owner (compose.yaml "user:"); nothing below needs root
EXPOSE 8095
CMD ["python", "-m", "seatsense.app"]
