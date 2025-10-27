FROM python:3.11-slim

# 替换 apt 源为阿里云镜像（Debian）
# COPY sources.list /etc/apt/sources.list

# Install OS deps needed by onnxruntime/opencv and build tools for fallback
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgomp1 \
       libstdc++6 \
       libglib2.0-0 \
       build-essential \
       cmake \
       git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install Python dependencies (prefer binary wheels; use tuna mirror with fallback)
COPY requirements.txt ./
RUN pip install --no-cache-dir --prefer-binary \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

# Copy source code
COPY api.py ./
COPY face_db.py ./
COPY data ./data

EXPOSE 8001

# Default command (compose will override with --reload)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]