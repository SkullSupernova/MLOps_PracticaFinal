# KMNIST CNN_ResNet — Pipeline MLOps

## Identificación y accesos del proyecto

* **Autor**: Miguel Angel Valbuena Bueno
* **Repositorio en GitHub**: [https://github.com/SkullSupernova/MLOps_PracticaFinal](https://github.com/SkullSupernova/MLOps_PracticaFinal)
* **Proyecto en Weights & Biases**: [https://wandb.ai/miguel-valbuena-bueno-/kmnist-resnet](https://wandb.ai/miguel-valbuena-bueno-/kmnist-resnet)

Pipeline completo de MLOps para clasificación de caracteres Kuzushiji mediante una red neuronal convolucional residual (`CNN_ResNet`) entrenada sobre el dataset KMNIST.

---

# Descripción del proyecto

El sistema implementa un modelo de clasificación de imágenes basado en arquitecturas residuales tipo ResNet para reconocer caracteres históricos japoneses del dataset KMNIST.

El proyecto integra:

* entrenamiento reproducible,
* evaluación automatizada,
* seguimiento de experimentos con Weights & Biases,
* inferencia mediante FastAPI,
* contenedorización con Docker,
* validación mediante tests y linting,
* automatización CI/CD.

---

# Características principales

* Arquitectura CNN residual optimizada para KMNIST
* Entrenamiento reproducible con checkpoints
* API REST mediante FastAPI
* Integración con Weights & Biases (W&B)
* Contenedorización con Docker
* Validación automática mediante pytest
* Linting y formateo con Ruff
* Evaluación OOD (Out-of-Distribution)
* CLI centralizada para operaciones del pipeline

---

# Arquitectura del proyecto

Versión reducida:

```text
src/
├── api.py            # Interfaz REST FastAPI
├── logging_config.py # Configuración de registros
├── main.py           # Orquestador CLI
├── model.py          # Arquitectura CNN_ResNet
├── train.py          # Pipeline de entrenamiento
└── utils.py          # Utilidades de datos
```

Versión extensa:

```text
MLOps_PracticaFinal/
├───data
│   └───KMNIST
│       └───raw
│               t10k-images-idx3-ubyte
│               t10k-images-idx3-ubyte.gz
│               t10k-labels-idx1-ubyte
│               t10k-labels-idx1-ubyte.gz
│               train-images-idx3-ubyte
│               train-images-idx3-ubyte.gz
│               train-labels-idx1-ubyte
│               train-labels-idx1-ubyte.gz
│
├───logs
│       errors.log
│       kmnist.log
│       project.log
│
├───models
│       ResNet_Final_Combined.pth
│
├───notebook
│       DL_Practica_final_25_26_MValbuena.ipynb
│       DL_Practica_final_25_26_Refactor.ipynb
│
├───src
│   │   api.py
│   │   logging_config.py
│   │   main.py
│   │   model.py
│   │   train.py
│   │   utils.py
│   │   __init__.py
│
├───test
│   │   conftest.py
│   │   test_api.py
│   │   test_model.py
│   │   test_utils.py
│
└───wandb
```

---

# Requisitos previos

## Software requerido

* Python 3.10 o superior
* PyTorch 2.1.0 o superior
* pip
* Git

## Opcional

* CUDA 11.8+ para aceleración GPU
* Docker y Docker Compose

---

# Instalación

## 1. Clonar el repositorio

```bash
git clone https://github.com/SkullSupernova/MLOps_PracticaFinal.git
cd MLOps_PracticaFinal
```

## 2. Crear entorno virtual

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 3. Instalar dependencias

```bash
pip install -e ".[dev]"
```

---

# Configuración del entorno

## Variables de entorno

Crear archivo `.env` a partir del ejemplo:

```bash
cp .env.example .env
```

## Variables disponibles

| Variable        | Descripción            | Valor por defecto                  |
| --------------- | ---------------------- | ---------------------------------- |
| `MODEL_PATH`    | Ruta del checkpoint    | `models/ResNet_Final_Combined.pth` |
| `DATASET_MEAN`  | Media de normalización | `0.1918`                           |
| `DATASET_STD`   | Desviación estándar    | `0.3483`                           |
| `WANDB_API_KEY` | API key de W&B         | Requerida para tracking            |

---

# Dataset

El proyecto utiliza el dataset KMNIST, compuesto por:

* 70.000 imágenes,
* escala de grises,
* resolución 28×28,
* 10 clases de caracteres Kuzushiji.

## Descarga manual

```bash
python -c "
import torchvision
torchvision.datasets.KMNIST(root='data', train=True, download=True)
torchvision.datasets.KMNIST(root='data', train=False, download=True)
"
```

Los datos se almacenan en:

```text
data/KMNIST/raw/
```

## Clases del dataset

| Índice | Hiragana | Romaji |
| ------ | -------- | ------ |
| 0      | お        | o      |
| 1      | き        | ki     |
| 2      | す        | su     |
| 3      | つ        | tsu    |
| 4      | な        | na     |
| 5      | は        | ha     |
| 6      | ま        | ma     |
| 7      | や        | ya     |
| 8      | れ        | re     |
| 9      | を        | wo     |

---

# Entrenamiento

## Entrenamiento estándar

```bash
python -m src.main train
```

El sistema reutiliza automáticamente checkpoints existentes si están disponibles.

## Reentrenamiento forzado

```bash
python -m src.main train --force-train
```

## Entrenamiento con W&B

```bash
python -m src.main train --use-wandb --max-epochs 100
```

---

# Evaluación e inferencia

## Evaluación sobre test set

```bash
python -m src.main evaluate \
  --checkpoint models/ResNet_Final_Combined.pth
```

## Evaluación OOD

```bash
python -m src.main ood \
  --checkpoint models/ResNet_Final_Combined.pth
```

## Verificación rápida del modelo

```bash
python -c "
import torch
from src.model import CNN_ResNet

model = CNN_ResNet().eval()
x = torch.randn(2, 1, 28, 28)
out = model(x)

print(out.shape)
"
```

Salida esperada:

```text
torch.Size([2, 10])
```

---

# API FastAPI

## Ejecución local

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints disponibles

| Método | Ruta          | Descripción             |
| ------ | ------------- | ----------------------- |
| GET    | `/health`     | Estado del servicio     |
| GET    | `/model/info` | Información del modelo  |
| POST   | `/predict`    | Inferencia sobre imagen |

## Documentación interactiva

Disponible en:

```text
http://localhost:8000/docs
```

## Ejemplo de inferencia

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@imagen.png"
```

## Verificación de estado

```bash
curl http://localhost:8000/health
```

---

# Docker

## Construcción de imagen

```bash
docker build -t kmnist-api:latest .
```

## Ejecución del contenedor

```bash
docker run --rm \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  kmnist-api:latest
```

## Docker Compose

```bash
docker-compose up --build
```

## Ver logs

```bash
docker-compose logs -f api
```

---

# Tests y calidad de código

## Tests unitarios

```bash
pytest test/ -v
```

## Cobertura

```bash
pytest test/ --cov=src --cov-report=term-missing
```

## Linting

```bash
ruff check src/ test/
```

## Formateo

```bash
ruff format src/ test/
```

---

# Weights & Biases (W&B)

## Configuración

Exportar la API key:

```bash
export WANDB_API_KEY=<tu_api_key>
```

## Ejecución con tracking

```bash
python -m src.main train --use-wandb
```

## Proyecto W&B

[Weights & Biases Project](https://wandb.ai/miguel-valbuena-bueno-/kmnist-resnet?utm_source=chatgpt.com)

---

## Integración y despliegue continuos (ci/cd)

El proyecto utiliza **GitHub Actions** para automatizar la validación del código mediante el flujo definido en `.github/workflows/ci.yml`. Cada envío de código (**push**) activa los siguientes procesos:

1. **Linting (ruff)**: garantiza la calidad y el cumplimiento de los estándares de estilo del código.
2. **Tests (pytest)**: ejecución de la suite de pruebas unitarias y de integración para validar la lógica del sistema.
3. **Validación de forward pass del modelo**: prueba funcional que confirma la integridad de la arquitectura **CNN_ResNet**.

---

# Resultados de referencia

| Métrica             | Valor   |
| ------------------- | ------- |
| Validation Accuracy | 98.62 % |
| Test F1 Macro       | ~0.986  |


---

# Validación rápida del sistema

## Verificar instalación

```bash
python -c "from src.model import CNN_ResNet; print('OK')"
```

## Ejecutar pipeline completo

```bash
python -m src.main train
python -m src.main evaluate
```

## Validar API

```bash
curl http://localhost:8000/health
```

---

# Troubleshooting

## Error: checkpoint no encontrado

Verificar existencia de:

```text
models/ResNet_Final_Combined.pth
```

---

## Error CUDA

Comprobar compatibilidad entre:

* PyTorch,
* CUDA,
* drivers NVIDIA.

---

## Error W&B authentication

Verificar:

```bash
echo $WANDB_API_KEY
```

---

## Error Docker mount

Verificar permisos sobre:

```text
models/
```