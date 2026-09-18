# Multi-stage enterprise production build for AlphaPulse Quantitative Terminal
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies into wheels layer
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final lean runtime stage
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged system user for enterprise container security
RUN groupadd -g 1001 quantgroup && \
    useradd -u 1001 -g quantgroup -s /bin/bash -m quantuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/quantuser/.local
ENV PATH=/home/quantuser/.local/bin:$PATH

# Copy application code
COPY . .
RUN chown -R quantuser:quantgroup /app

USER quantuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
