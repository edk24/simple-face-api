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
       wget \
       unzip \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app


# 设置 InsightFace 模型缓存目录
ENV INSIGHTFACE_HOME=/app/models

# Install Python dependencies (prefer binary wheels; use tuna mirror with fallback)
COPY requirements.txt ./
RUN pip install --no-cache-dir --prefer-binary \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt


# 预下载 InsightFace buffalo_l 模型
RUN mkdir -p /app/models/buffalo_l \
    && wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -O /tmp/buffalo_l.zip \
    && unzip -q /tmp/buffalo_l.zip -d /app/models/buffalo_l/ \
    && rm /tmp/buffalo_l.zip

# Copy source code
COPY api.py ./
COPY face_db.py ./
COPY data ./data

EXPOSE 8001

# Default command (compose will override with --reload)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]