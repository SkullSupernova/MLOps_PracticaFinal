# =============================================================================
# api.py
# API REST de inferencia para el modelo KMNIST CNN_ResNet.
#
# Endpoints:
#   GET  /health           — Estado del servicio y del modelo.
#   POST /predict          — Inferencia sobre imagen PNG/JPG (28x28 o reescalada).
#   GET  /model/info       — Metadatos del modelo cargado.
#
# Ejecución:
#   uvicorn src.api:app --host 0.0.0.0 --port 8000
# =============================================================================

import io
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src.logging_config import get_logger, setup_logging
from src.model import CNN_ResNet
from src.utils import load_checkpoint

setup_logging()
logger = get_logger(__name__)

# =============================================================================
# Configuración
# =============================================================================
CLASS_NAMES    = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']
DEFAULT_CKPT   = os.getenv(
    'MODEL_PATH',
    str(Path(__file__).parent.parent / 'models' / 'ResNet_Final_Combined.pth'),
)
MEAN_VAL       = float(os.getenv('DATASET_MEAN', '0.1918'))
STD_VAL        = float(os.getenv('DATASET_STD',  '0.3483'))

# =============================================================================
# Estado global del servicio
# =============================================================================
_state: dict = {
    'model':   None,
    'device':  None,
    'metrics': {},
    'transform': None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar el servidor y lo libera al detenerse."""
    logger.info("Cargando modelo desde: %s", DEFAULT_CKPT)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = CNN_ResNet().to(device)

    if not Path(DEFAULT_CKPT).exists():
        logger.error("Archivo de checkpoint no encontrado: %s", DEFAULT_CKPT)
    else:
        success, _, metrics = load_checkpoint(DEFAULT_CKPT, model, device)
        if success:
            model.eval()
            _state['model']   = model
            _state['metrics'] = metrics
            logger.info("Modelo cargado. Val Acc: %s", metrics.get('val_acc', 'N/A'))
        else:
            logger.error("Fallo al cargar el checkpoint.")

    _state['device'] = device
    _state['transform'] = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((MEAN_VAL,), (STD_VAL,)),
    ])

    yield

    logger.info("Servicio detenido. Liberando recursos.")
    _state['model'] = None


# =============================================================================
# Aplicación
# =============================================================================
app = FastAPI(
    title="KMNIST CNN_ResNet Inference API",
    description="Clasificación de caracteres Kuzushiji mediante CNN ResNet.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# =============================================================================
# Esquemas de respuesta
# =============================================================================
class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    device:       str
    val_acc:      Optional[float] = None


class PredictionResponse(BaseModel):
    predicted_class:  str
    class_index:      int
    confidence:       float
    probabilities:    dict[str, float]


class ModelInfoResponse(BaseModel):
    architecture:    str
    num_parameters:  int
    num_classes:     int
    class_names:     list[str]
    val_acc:         Optional[float] = None
    val_loss:        Optional[float] = None
    checkpoint_path: str


# =============================================================================
# Endpoints
# =============================================================================
@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health():
    """Comprueba el estado del servicio y confirma si el modelo está cargado."""
    model_loaded = _state['model'] is not None
    val_acc      = _state['metrics'].get('val_acc') if model_loaded else None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        device=str(_state['device']),
        val_acc=val_acc,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Modelo"])
async def model_info():
    """Devuelve los metadatos del modelo cargado."""
    model = _state.get('model')
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo no está disponible.",
        )
    n_params = sum(p.numel() for p in model.parameters())
    return ModelInfoResponse(
        architecture="CNN_ResNet",
        num_parameters=n_params,
        num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        val_acc=_state['metrics'].get('val_acc'),
        val_loss=_state['metrics'].get('val_loss'),
        checkpoint_path=DEFAULT_CKPT,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inferencia"])
async def predict(file: UploadFile = File(...)):
    """
    Realiza la inferencia sobre una imagen de carácter Kuzushiji.

    Acepta imágenes PNG o JPEG. Si las dimensiones no son 28x28,
    se reescala automáticamente.

    Retorna la clase predicha, el índice, la confianza y el vector
    completo de probabilidades Softmax.
    """
    model = _state.get('model')
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo no está disponible.",
        )

    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Tipo de archivo no soportado: {file.content_type}. "
                   "Use PNG o JPEG.",
        )

    try:
        contents = await file.read()
        img      = Image.open(io.BytesIO(contents)).convert("L")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No se pudo decodificar la imagen: {exc}",
        )

    tensor = _state['transform'](img).unsqueeze(0).to(_state['device'])

    with torch.no_grad():
        outputs  = model(tensor)
        probs    = F.softmax(outputs, dim=1).cpu().squeeze().numpy()
        pred_idx = int(np.argmax(probs))

    return PredictionResponse(
        predicted_class=CLASS_NAMES[pred_idx],
        class_index=pred_idx,
        confidence=float(probs[pred_idx]),
        probabilities={name: float(p) for name, p in zip(CLASS_NAMES, probs)},
    )