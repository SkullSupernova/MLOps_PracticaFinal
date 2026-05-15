# KMNIST CNN_ResNet — Pipeline MLOps

Clasificación de caracteres Kuzushiji (KMNIST) mediante una red neuronal convolucional
residual (CNN_ResNet) con pipeline MLOps completo: entrenamiento reproducible, API REST
de inferencia, Dockerización y CI/CD con GitHub Actions.

## Arquitectura del proyecto

    src/
    ├── api.py            # API REST FastAPI
    ├── logging_config.py # Logging centralizado
    ├── main.py           # CLI principal
    ├── model.py          # Arquitectura CNN_ResNet
    ├── train.py          # Pipeline de entrenamiento
    └── utils.py          # Utilidades de datos y visualización

## Requisitos

- Python 3.10 o superior
- PyTorch 2.1.0 o superior
- CUDA 11.8+ (opcional, para aceleración GPU)

## Instalación

    git clone https://github.com/<usuario>/kmnist-resnet.git
    cd kmnist-resnet
    python -m venv .venv
    source .venv/bin/activate          # Linux/macOS
    # .venv\Scripts\activate           # Windows
    pip install -e ".[dev]"

Copiar y rellenar las variables de entorno:

    cp .env.example .env

## Uso

### Verificar versiones del entorno

    python -m src.main --version

### Entrenamiento

    # Respeta checkpoint previo si existe
    python -m src.main train

    # Forzar reentrenamiento desde cero
    python -m src.main train --force-train

    # Con integración W&B
    python -m src.main train --use-wandb --max-epochs 100

### Evaluación sobre el test set

    python -m src.main evaluate --checkpoint models/ResNet_Final_Combined.pth

### Evaluación de robustez OOD

    python -m src.main ood --checkpoint models/ResNet_Final_Combined.pth

## API REST

### Ejecución local

    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

### Endpoints

| Método | Ruta         | Descripción                              |
|--------|--------------|------------------------------------------|
| GET    | /health      | Estado del servicio y del modelo         |
| GET    | /model/info  | Metadatos del modelo cargado             |
| POST   | /predict     | Inferencia sobre imagen PNG/JPEG         |

Documentación interactiva disponible en: http://localhost:8000/docs

### Ejemplo de inferencia

    curl -X POST http://localhost:8000/predict \
         -F "file=@ruta/imagen.png"

## Docker

### Construcción

    docker build -t kmnist-api:latest .

### Ejecución

    docker run --rm -p 8000:8000 \
      -e WANDB_API_KEY=<clave> \
      -v $(pwd)/models:/app/models:ro \
      kmnist-api:latest

### Docker Compose

    docker-compose up --build

## Tests

    # Ejecutar todos los tests
    pytest test/ -v

    # Con informe de cobertura
    pytest test/ --cov=src --cov-report=term-missing

## Linting

    ruff check src/ test/
    ruff format src/ test/

## Dataset

KMNIST contiene 70.000 imágenes en escala de grises (28x28) de 10 caracteres
Kuzushiji (escritura japonesa cursiva histórica).

Los datos se esperan en: `data/KMNIST/raw/`

Para descargar manualmente:

    python -c "
    import torchvision
    torchvision.datasets.KMNIST(root='data', train=True,  download=True)
    torchvision.datasets.KMNIST(root='data', train=False, download=True)
    "

## Clases

| Índice | Hiragana | Romaji |
|--------|----------|--------|
| 0      | お       | o      |
| 1      | き       | ki     |
| 2      | す       | su     |
| 3      | つ       | tsu    |
| 4      | な       | na     |
| 5      | は       | ha     |
| 6      | ま       | ma     |
| 7      | や       | ya     |
| 8      | れ       | re     |
| 9      | を       | wo     |

## Resultados de referencia

| Métrica      | Valor   |
|--------------|---------|
| Val Accuracy | 98.62 % |
| Test F1 macro| ~0.986  |

## Variables de entorno

| Variable        | Descripción                          | Valor por defecto                       |
|-----------------|--------------------------------------|-----------------------------------------|
| MODEL_PATH      | Ruta al checkpoint .pth              | models/ResNet_Final_Combined.pth        |
| DATASET_MEAN    | Media de normalización               | 0.1918                                  |
| DATASET_STD     | Desviación estándar de normalización | 0.3483                                  |
| WANDB_API_KEY   | Clave API de W&B                     | (obligatoria si se usa --use-wandb)     |

# Comandos de ejecución y validación

# Entorno local

## 1. Instalar paquete en modo editable
pip install -e ".[dev]"

## 2. Verificar instalación
python -c "from src.model import CNN_ResNet; print('OK')"

## 3. Verificar forward pass
python -c "
import torch
from src.model import CNN_ResNet
m = CNN_ResNet().eval()
x = torch.randn(2, 1, 28, 28)
out = m(x)
print('Shape de salida:', out.shape)   # debe ser torch.Size([2, 10])
"

## 4. Ejecutar tests completos
pytest test/ -v --cov=src

## 5. Linting
ruff check src/ test/

## 6. Entrenamiento (respeta checkpoint existente)
python -m src.main train

## 7. Entrenamiento forzado
python -m src.main train --force-train --max-epochs 100

## 8. Evaluación
python -m src.main evaluate --checkpoint models/ResNet_Final_Combined.pth

## 9. OOD
python -m src.main ood --checkpoint models/ResNet_Final_Combined.pth

## 10. API local
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

## 11. Probar endpoint health
curl http://localhost:8000/health

## 12. Inferencia de ejemplo (requiere imagen PNG en el directorio actual)
curl -X POST http://localhost:8000/predict -F "file=@imagen_ejemplo.png"

## 13. Información del modelo
curl http://localhost:8000/model/info

# Docker

## Construcción de la imagen
docker build -t kmnist-api:latest .

## Ejecución del contenedor
docker run --rm -p 8000:8000 \
  -v "$(pwd)/models:/app/models:ro" \
  kmnist-api:latest

## Verificación del health check del contenedor
docker inspect kmnist_inference | grep -A5 '"Health"'

## Ejecución con docker-compose
docker-compose up --build -d

## Ver logs
docker-compose logs -f api