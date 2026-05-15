# =============================================================================
# Dockerfile
# Imagen de producción para la API de inferencia KMNIST CNN_ResNet.
# Build en dos etapas: builder (instala dependencias) + runtime (imagen limpia).
# =============================================================================

# --- Etapa 1: builder ---
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml requirements.txt ./

# Generación de entorno virtual aislado
RUN python -m venv /opt/venv
# Exposición del entorno virtual en el PATH
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Etapa 2: runtime ---
FROM python:3.11-slim AS runtime

LABEL maintainer="kmnist-project"
LABEL description="API de inferencia KMNIST CNN_ResNet"

# Creación de usuario sin privilegios por seguridad
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Transferencia del entorno virtual completo desde la etapa constructora
COPY --from=builder /opt/venv /opt/venv

# Exposición del entorno virtual en el PATH del contenedor final
ENV PATH="/opt/venv/bin:$PATH"

# Copia de código fuente y configuraciones
COPY src/     ./src/
COPY models/  ./models/
COPY config/  ./config/

# Ajuste de permisos para escritura de registros
RUN mkdir -p logs && chown -R appuser:appuser /app

USER appuser

# Puerto de exposición
EXPOSE 8000

# Variables de entorno por defecto
ENV MODEL_PATH=/app/models/ResNet_Final_Combined.pth \
    DATASET_MEAN=0.1918 \
    DATASET_STD=0.3483 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check integrado
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Ejecución del servicio mediante el ejecutable del entorno virtual
# Ejecución absoluta garantizada sin depender del PATH dinámico
CMD ["/opt/venv/bin/python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]