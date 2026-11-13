# CPU-only inference + evaluation image. No network access is needed at runtime.
#   docker build -t echogesture .
#   docker run --rm -v /path/to/test_clips:/clips -v $PWD/out:/out echogesture \
#       python scripts/make_predictions.py --clips-dir /clips --out /out/predictions.jsonl
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY echogesture/ echogesture/
COPY scripts/ scripts/
COPY demo/ demo/
COPY assets/ assets/
COPY models/ models/
COPY predict.py .

ENV GLOG_minloglevel=2
CMD ["python", "predict.py"]
