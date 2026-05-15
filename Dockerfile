# =============================================================================
# Dockerfile
# Imagen de producción para la API de inferencia KMNIST CNN_ResNet.
# Build en dos etapas: builder (instala dependencias) + runtime (imagen limpia).
# =============================================================================

# --- Etapa 1: builder ---
FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Etapa 2: runtime ---
FROM python:3.11-slim AS runtime

LABEL maintainer="kmnist-project"
LABEL description="API de inferencia KMNIST CNN_ResNet"

# Usuario sin privilegios (seguridad)
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Copiar dependencias instaladas desde builder
COPY --from=builder /install /usr/local

# Copiar código fuente
COPY src/     ./src/
COPY models/  ./models/
COPY config/  ./config/

# Crear directorio de logs con permisos adecuados
RUN mkdir -p logs && chown -R appuser:appuser /app

USER appuser

# Puerto de exposición
EXPOSE 8000

# Variables de entorno por defecto (sobreescribibles en docker run)
ENV MODEL_PATH=/app/models/ResNet_Final_Combined.pth \
    DATASET_MEAN=0.1918 \
    DATASET_STD=0.3483 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check integrado
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]