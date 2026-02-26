# Práctica Hands-On: MLOps con MLflow — Entorno Dockerizado en GCP
## Pipeline End-to-End de Clasificación con Tracking, Registry y Serving

---

## Introducción y Objetivos

### Objetivo del Lab

Construir un pipeline MLOps completo usando **MLflow** en un entorno **completamente dockerizado** sobre Compute Engine (4 GB - 8 GB RAM) que incluya:

- ✅ MLflow Tracking Server con GUI en **puerto 80** (Nginx reverse proxy)
- ✅ PostgreSQL dockerizado como backend store
- ✅ GCS como artifact store
- ✅ MLflow Model Registry para versionado
- ✅ Model Serving como contenedor independiente
- ✅ Pipeline automatizado de entrenamiento
- ✅ Monitoring básico de modelos
- ✅ **uv** como gestor de entorno Python (ultra-rápido)

### Dataset: Wine Quality Classification

Usaremos el **Wine Quality Dataset** de UCI:
- **Objetivo**: Clasificar calidad de vino (binary: good/bad wine)
- **Features**: 11 atributos fisicoquímicos (acidez, azúcar, alcohol, etc.)
- **Tamaño**: ~4,900 ejemplos
- **Tipo**: Clasificación binaria
- **Razón**: problema real de negocio

### Arquitectura Dockerizada


<img width="1163" height="833" alt="imagen" src="https://github.com/user-attachments/assets/8ccaa950-ede5-41c3-ba10-ef0881c9edfb" />













## Prerequisitos y Setup de VM

### Requisitos
- Cuenta Google Cloud con billing habilitado
- Proyecto GCP creado
- Permisos: Compute Engine Admin, Storage Admin

### Costes Estimados
**VM Compute Engine**:
- Tipo: e2-standard-2 (2 vCPUs, **8 GB RAM**)
- Coste: ~$49/mes (~$0.067/hora)
- Storage: 30 GB SSD (suficiente, artifacts van a GCS)


### Crear Infraestructura GCP

```bash
# Variables de configuración
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"
export ZONE="us-central1-a"
export VM_NAME="mlflow-server"
export BUCKET_NAME="${PROJECT_ID}-mlflow-artifacts"

# Crear bucket para artifacts
gcloud storage buckets create gs://$BUCKET_NAME \
    --location=$REGION \
    --uniform-bucket-level-access

# Regla de firewall: solo puerto 80 (HTTP estándar)
gcloud compute firewall-rules create allow-mlflow-http \
    --direction=INGRESS \
    --priority=1000 \
    --network=default \
    --action=ALLOW \
    --rules=tcp:80 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=mlflow-server

# Obtener IP externa
export VM_EXTERNAL_IP=$(gcloud compute instances describe $VM_NAME \
    --zone=$ZONE \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo "VM creada en IP: $VM_EXTERNAL_IP"
echo "MLflow UI estará disponible en: http://$VM_EXTERNAL_IP"
```

### Conectar y Preparar la VM

```bash
# SSH a la VM
gcloud compute ssh $VM_NAME --zone=$ZONE
```

Una vez dentro de la VM, instalar Docker y uv:

```bash
# ─── Instalar Docker ───
# Dependencias
sudo apt-get update
sudo apt install docker
sudo apt-get install docker-compose -y

# Permitir uso sin sudo
sudo usermod -aG docker $USER

# Iniciar docker
sudo systemctl start docker
sudo systemctl enable docker

# Verificar
docker --version
docker-compose version

# ─── Instalar uv ───
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Verificar
uv --version
```

**¿Por qué uv?**:
- Resolución de dependencias 10-100x más rápido que pip
- Drop-in replacement para pip y venv
- Ideal para entornos con recursos limitados (menos CPU/RAM durante instalación)
- **Se usa solo en el host** para los scripts de training/pipeline, no dentro de los contenedores Docker (que usan pip estándar)

---

## Parte 1: Estructura del Proyecto

### 1.1 Crear Estructura de Directorios

```bash
# Crear directorio raíz del proyecto
mkdir -p ~/mlflow-project
cd ~/mlflow-project

# Estructura completa
mkdir -p \
    docker/postgres \
    docker/mlflow \
    docker/nginx \
    docker/serving \
    scripts \
    data

# Vista final esperada:
# mlflow-project/
# ├── docker-compose.yml
# ├── docker/
# │   ├── postgres/
# │   │   └── init.sql
# │   ├── mlflow/
# │   │   ├── Dockerfile
# │   │   └── entrypoint.sh
# │   ├── nginx/
# │   │   └── nginx.conf
# │   └── serving/
# │       ├── Dockerfile
# │       └── entrypoint.sh
# ├── scripts/
# │   ├── prepare_data.py
# │   ├── train_model.py
# │   ├── query_experiments.py
# │   ├── register_model.py
# │   ├── manage_model_versions.py
# │   ├── predict_client.py
# │   ├── mlops_pipeline.py
# │   ├── monitor_model.py
# │   └── alert_system.py
# ├── data/
# └── pyproject.toml
```

---

## Parte 2: Configuración Docker

### 2.1 PostgreSQL — Script de Inicialización

```bash
cat > ~/mlflow-project/docker/postgres/init.sql <<'EOF'
-- Crear extensiones necesarias
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Verificar BD
SELECT version();
SELECT current_database();
EOF
```

**Explicación**: Docker Postgres ejecuta automáticamente scripts en `/docker-entrypoint-initdb.d/` en el primer arranque. La BD y el usuario se crean vía variables de entorno directamente en el `docker-compose.yml`.

### 2.2 MLflow Server — Dockerfile

```bash
cat > ~/mlflow-project/docker/mlflow/Dockerfile <<'EOF'
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==2.18.0 \
    psycopg2-binary==2.9.10 \
    google-cloud-storage==2.19.0 \
    gunicorn==23.0.0

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
EOF
```

**Optimización para 4 GB**:
- `python:3.11-slim`: Imagen base ~150 MB vs ~900 MB de la completa
- `--no-cache-dir` en pip: No almacena cache de paquetes, ahorra ~200 MB
- `--workers 2 --worker-class gthread --threads 2`: 2 workers con 2 threads cada uno = 4 requests concurrentes, consumo ~300 MB total
- `--no-install-recommends`: Solo paquetes esenciales del sistema

### 2.3 MLflow Server — Entrypoint Script

> **Paso crítico**: El Dockerfile del paso anterior hace `COPY entrypoint.sh /entrypoint.sh`. Es necesario crear este script en el mismo directorio (`docker/mlflow/`) antes de construir la imagen.

```bash
cat > ~/mlflow-project/docker/mlflow/entrypoint.sh <<'SCRIPT'
#!/bin/bash
set -e

# ─── Auto-detectar GCP Project ID ───
# Las credenciales tipo "authorized_user" (generadas con gcloud auth application-default login)
# NO incluyen el Project ID, a diferencia de las service account keys.
# La librería google-cloud-storage lo necesita para operar con GCS.
# Este bloque lo extrae automáticamente del campo "quota_project_id" o "project_id"
# del fichero de credenciales, evitando tener que hardcodear el Project ID en el
# docker-compose.yml o en variables de entorno.
if [ -z "$GCLOUD_PROJECT" ]; then
    export GCLOUD_PROJECT=$(python3 -c "
import json, os
cred_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
if cred_file and os.path.exists(cred_file):
    with open(cred_file) as f:
        creds = json.load(f)
    print(creds.get('quota_project_id', creds.get('project_id', '')))
" 2>/dev/null)
    export GOOGLE_CLOUD_PROJECT="$GCLOUD_PROJECT"
    echo "  Auto-detected GCP Project: ${GCLOUD_PROJECT}"
fi

echo "Starting MLflow Tracking Server..."
echo "  Backend: ${MLFLOW_BACKEND_URI}"
echo "  Artifacts: ${MLFLOW_ARTIFACT_ROOT}"
echo "  Project: ${GCLOUD_PROJECT}"

exec mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_URI}" \
    --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 2 \
    --gunicorn-opts "--worker-class gthread --threads 2 --timeout 120"
SCRIPT

chmod +x ~/mlflow-project/docker/mlflow/entrypoint.sh
```

**Explicación**:
- `set -e`: El script falla inmediatamente si cualquier comando da error
- **Auto-detección de Project ID**: Las credenciales tipo `authorized_user` (generadas con `gcloud auth application-default login`) no incluyen el Project ID automáticamente, a diferencia de una service account key. La librería `google-cloud-storage` lo necesita y lo busca en `GCLOUD_PROJECT` / `GOOGLE_CLOUD_PROJECT`. Este bloque lo extrae del campo `quota_project_id` del fichero de credenciales, haciendo el setup portable sin hardcodear nada
- `exec`: Reemplaza el proceso del shell con MLflow, para que las señales de Docker (stop/kill) lleguen directamente al servidor
- `--workers 2 --worker-class gthread --threads 2`: 2 workers con 2 threads = 4 requests concurrentes, equilibrio óptimo para 8 GB RAM
- `--timeout 120`: Evita que gunicorn mate workers durante operaciones lentas (subida de artifacts grandes)
- Las variables `MLFLOW_BACKEND_URI` y `MLFLOW_ARTIFACT_ROOT` se inyectan desde `docker-compose.yml`

### 2.4 Nginx — Reverse Proxy

```bash
cat > ~/mlflow-project/docker/nginx/nginx.conf <<'EOF'
worker_processes 1;
worker_rlimit_nofile 1024;

events {
    worker_connections 512;
}

http {
    client_body_buffer_size 16k;
    client_max_body_size 50m;
    proxy_buffer_size 8k;
    proxy_buffers 4 16k;

    proxy_connect_timeout 60s;
    proxy_send_timeout 120s;
    proxy_read_timeout 120s;

    gzip on;
    gzip_types text/plain application/json text/css application/javascript;

    upstream mlflow_ui {
        server mlflow-server:5000;
    }

    server {
        listen 80;
        server_name _;

        # Resolver de Docker para resolución dinámica
        resolver 127.0.0.11 valid=30s ipv6=off;

        # MLflow UI y API
        location / {
            proxy_pass http://mlflow_ui;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Model Serving API — resolución dinámica (no falla si el contenedor no existe)
        location /api/predict {
            set $serving_upstream http://mlflow-serving:5001;
            rewrite ^/api/predict(.*) /invocations$1 break;
            proxy_pass $serving_upstream;
            proxy_set_header Host $host;
            proxy_set_header Content-Type $content_type;
        }

        location /health/mlflow {
            proxy_pass http://mlflow_ui/health;
        }

        location /health/serving {
            set $serving_upstream http://mlflow-serving:5001;
            proxy_pass $serving_upstream/health;
        }

        location ~ /\. {
            deny all;
        }
    }
}
EOF
```

**Explicación de Nginx**:
- **Puerto 80**: Estándar HTTP, sin necesidad de especificar puerto en el navegador
- `/` → MLflow UI (puerto 5000 interno)
- `/api/predict` → Model Serving (reescribe a `/invocations`)
- `/health/*` → Health checks de cada servicio
- `worker_processes 1`: Suficiente para este caso, ahorra RAM
- `gzip on`: Comprime respuestas, mejora velocidad de la UI

### 2.5 Model Serving — Dockerfile

```bash
cat > ~/mlflow-project/docker/serving/Dockerfile <<'EOF'
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==2.18.0 \
    psycopg2-binary==2.9.10 \
    google-cloud-storage==2.19.0 \
    scikit-learn==1.5.2 \
    pandas==2.2.3 \
    numpy==1.26.4

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
EOF
```

```bash
cat > ~/mlflow-project/docker/serving/entrypoint.sh <<'SCRIPT'
#!/bin/bash
set -e

# ─── Auto-detectar GCP Project ID (mismo mecanismo que el MLflow server) ───
if [ -z "$GCLOUD_PROJECT" ]; then
    export GCLOUD_PROJECT=$(python3 -c "
import json, os
cred_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
if cred_file and os.path.exists(cred_file):
    with open(cred_file) as f:
        creds = json.load(f)
    print(creds.get('quota_project_id', creds.get('project_id', '')))
" 2>/dev/null)
    export GOOGLE_CLOUD_PROJECT="$GCLOUD_PROJECT"
    echo "  Auto-detected GCP Project: ${GCLOUD_PROJECT}"
fi

echo "Starting MLflow Model Serving..."
echo "  Model: models:/${MODEL_NAME}/Production"
echo "  Tracking URI: ${MLFLOW_TRACKING_URI}"
echo "  Project: ${GCLOUD_PROJECT}"

exec mlflow models serve \
    --model-uri "models:/${MODEL_NAME}/Production" \
    --host 0.0.0.0 \
    --port 5001 \
    --no-conda \
    --env-manager=local
SCRIPT

chmod +x ~/mlflow-project/docker/serving/entrypoint.sh
```

### 2.6 Docker Compose

```bash
cat > ~/mlflow-project/docker-compose.yml <<'COMPOSEEOF'
services:

  # ─── PostgreSQL ───
  postgres:
    image: postgres:16-alpine
    container_name: mlflow-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow_secure_pwd_2024
      POSTGRES_DB: mlflow_db
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./docker/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    ports:
      - "127.0.0.1:5432:5432"
    networks:
      - mlflow-net
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mlflow -d mlflow_db"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 20s
    # Optimización PostgreSQL para 8 GB RAM
    command: >
      postgres
      -c shared_buffers=64MB
      -c work_mem=4MB
      -c maintenance_work_mem=32MB
      -c effective_cache_size=256MB
      -c max_connections=20
      -c wal_buffers=4MB
      -c checkpoint_completion_target=0.9
      -c random_page_cost=1.1
    shm_size: '256mb'
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  # ─── MLflow Tracking Server ───
  mlflow-server:
    build:
      context: ./docker/mlflow
      dockerfile: Dockerfile
    container_name: mlflow-tracking
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      MLFLOW_BACKEND_URI: postgresql://mlflow:mlflow_secure_pwd_2024@postgres:5432/mlflow_db
      MLFLOW_ARTIFACT_ROOT: gs://YOUR_PROJECT_ID-mlflow-artifacts/mlflow-artifacts
      GOOGLE_APPLICATION_CREDENTIALS: /run/secrets/gcp-credentials
    volumes:
      - ${HOME}/.config/gcloud/application_default_credentials.json:/run/secrets/gcp-credentials:ro
    networks:
      - mlflow-net
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  # ─── Nginx Reverse Proxy ───
  nginx:
    image: nginx:alpine
    container_name: mlflow-nginx
    restart: unless-stopped
    depends_on:
      - mlflow-server
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
    networks:
      - mlflow-net
    deploy:
      resources:
        limits:
          memory: 64M
        reservations:
          memory: 32M

  # ─── Model Serving (perfil separado, se levanta on-demand) ───
  mlflow-serving:
    build:
      context: ./docker/serving
      dockerfile: Dockerfile
    container_name: mlflow-serving
    restart: unless-stopped
    depends_on:
      - mlflow-server
    environment:
      MLFLOW_TRACKING_URI: http://mlflow-server:5000
      MODEL_NAME: wine-quality-random_forest-best
      GOOGLE_APPLICATION_CREDENTIALS: /run/secrets/gcp-credentials
    volumes:
      - ${HOME}/.config/gcloud/application_default_credentials.json:/run/secrets/gcp-credentials:ro
    networks:
      - mlflow-net
    profiles:
      - serving
    deploy:
      resources:
        limits:
          memory: 768M
        reservations:
          memory: 256M

networks:
  mlflow-net:
    driver: bridge

volumes:
  pgdata:
    driver: local
COMPOSEEOF

# ─── Reemplaza automáticamente YOUR_PROJECT_ID con tu proyecto real ───
sed -i "s/YOUR_PROJECT_ID/$PROJECT_ID/g" ~/mlflow-project/docker-compose.yml
```

**Explicación del Docker Compose**:

| Servicio | Recurso RAM | Descripción |
|----------|-------------|-------------|
| `postgres` | 256-512 MB | Base de datos con parámetros optimizados para poca RAM |
| `mlflow-server` | 256-512 MB | 2 gunicorn workers con threads |
| `nginx` | 32-64 MB | Reverse proxy ligero |
| `mlflow-serving` | 256-768 MB | Solo se levanta cuando hay modelo en Production |

### 2.7 Configurar Credenciales GCP

```bash
# Generar credenciales de aplicación (necesario para que los contenedores accedan a GCS)
gcloud auth application-default login

# Verificar que existe el archivo
ls -la ~/.config/gcloud/application_default_credentials.json

# Test de acceso al bucket
gcloud storage ls gs://$BUCKET_NAME/
```

**Explicación**: Los contenedores montan las credenciales del host como volumen read-only. Esto es más seguro que copiar la service account key dentro de la imagen Docker.

### 2.8 Levantar el Entorno

```bash
cd ~/mlflow-project

# Construir imágenes
docker-compose build

# Levantar servicios base (sin serving aún)
docker-compose up -d

# Verificar que todo está corriendo
docker-compose ps
# Output esperado:
# NAME              IMAGE                          STATUS                    PORTS
# mlflow-nginx      nginx:alpine                   Up 10 seconds             0.0.0.0:80->80/tcp
# mlflow-tracking   mlflow-project-mlflow-server   Up 10 seconds (healthy)   5000/tcp
# mlflow-postgres   postgres:16-alpine             Up 15 seconds (healthy)   127.0.0.1:5432->5432/tcp

# Verificar logs
docker-compose logs -f mlflow-server

# Test de salud
curl http://localhost/health/mlflow
# Output: {"status": "OK"}

# Ver consumo de recursos
docker stats --no-stream
```

**Acceso al UI**: http://<VM_EXTERNAL_IP>

Abrir en navegador — Se verá el MLflow UI en el **puerto 80** (sin necesidad de especificar puerto).

### 2.9 Setup del Entorno Python en el Host (con uv)

Los scripts de training, pipeline y monitoring se ejecutan en el host (no en contenedores) para tener acceso directo a GPU si la hubiera y para simplificar el desarrollo iterativo.

```bash
cd ~/mlflow-project

# Crear pyproject.toml
cat > pyproject.toml <<'EOF'
[project]
name = "mlflow-wine-quality"
version = "1.0.0"
description = "MLOps pipeline con MLflow - Wine Quality Classification"
requires-python = ">=3.11"

dependencies = [
    "mlflow==2.18.0",
    "psycopg2-binary==2.9.10",
    "google-cloud-storage==2.19.0",
    "scikit-learn==1.5.2",
    "pandas==2.2.3",
    "numpy==1.26.4",
    "matplotlib==3.9.3",
    "seaborn==0.13.2",
]
EOF

# Crear venv con uv (mucho más rápido que python -m venv + pip install)
uv venv .venv --python 3.11

# Activar
source .venv/bin/activate

# Instalar dependencias (uv resuelve e instala en segundos)
uv pip install -r pyproject.toml

# Verificar
python -c "import mlflow; print(f'MLflow {mlflow.__version__}')"
mlflow --version
```

**Comparación uv vs pip**:
```
# pip install (versión anterior): 45-90 segundos
# uv pip install: 3-8 segundos
# Speedup: ~10-15x
```

### 2.9 Variables de Entorno para Scripts del Host

```bash
# Añadir al ~/.bashrc
cat >> ~/.bashrc <<'ENVBLOCK'
# ─── MLflow Project ───
export MLFLOW_TRACKING_URI="http://localhost:80"
export MLFLOW_EXPERIMENT_NAME="wine-quality-classification"
export BUCKET_NAME="YOUR_PROJECT_ID-mlflow-artifacts"
cd ~/mlflow-project && source .venv/bin/activate 2>/dev/null
ENVBLOCK

# Reemplazar con tu PROJECT_ID real
sed -i "s/YOUR_PROJECT_ID/$PROJECT_ID/g" ~/.bashrc

# Cargar ahora
source ~/.bashrc

# Verificar conectividad con MLflow
mlflow experiments search
```

**Nota importante**: Los scripts del host usan `http://localhost:80` (a través de Nginx), mientras que los contenedores usan `http://mlflow-server:5000` (red Docker interna).

---

## Parte 3: Preparación de Datos

### 3.1 Descargar Wine Quality Dataset

```bash
cd ~/mlflow-project

# Descargar dataset
curl -o data/winequality-red.csv \
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

curl -o data/winequality-white.csv \
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv

# Verificar descarga
wc -l data/winequality-*.csv
# winequality-red.csv: 1,600 lines
# winequality-white.csv: 4,899 lines
```

### 3.2 Script de Preparación de Datos

```python
# Archivo: scripts/prepare_data.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import subprocess

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')


def load_wine_data():
    """Carga y combina datasets de vino tinto y blanco"""
    red_wine = pd.read_csv(os.path.join(DATA_DIR, 'winequality-red.csv'), sep=';')
    white_wine = pd.read_csv(os.path.join(DATA_DIR, 'winequality-white.csv'), sep=';')

    red_wine['wine_type'] = 0  # Red
    white_wine['wine_type'] = 1  # White

    wine_data = pd.concat([red_wine, white_wine], ignore_index=True)

    print(f"Total samples: {len(wine_data)}")
    print(f"Red wine: {len(red_wine)}, White wine: {len(white_wine)}")

    return wine_data


def create_quality_binary(df):
    """
    Convierte quality (0-10) a clasificación binaria:
    - quality >= 7: good wine (1)
    - quality < 7: bad wine (0)
    """
    df['quality_binary'] = (df['quality'] >= 7).astype(int)

    print("\nDistribución de calidad:")
    print(df['quality'].value_counts().sort_index())
    print(f"\nClase positiva (good wine): {df['quality_binary'].sum()}")
    print(f"Clase negativa (bad wine): {(1 - df['quality_binary']).sum()}")
    print(f"Balance: {df['quality_binary'].mean():.2%}")

    return df


def prepare_features(df):
    """Prepara features para modelo"""
    feature_cols = [col for col in df.columns if col not in ['quality', 'quality_binary']]

    X = df[feature_cols]
    y = df['quality_binary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {list(feature_cols)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_processed_data(X_train, X_test, y_train, y_test, scaler):
    """Guarda datos procesados localmente y sube a GCS"""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    X_train.to_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'), index=False, header=True)
    y_test.to_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'), index=False, header=True)

    with open(os.path.join(PROCESSED_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print("\n✓ Datos procesados guardados en data/processed/")

    # Subir a GCS
    bucket_name = os.environ.get('BUCKET_NAME')
    if bucket_name:
        subprocess.run(
            ['gcloud', 'storage', 'cp', '-r',
             f'{PROCESSED_DIR}/*',
             f'gs://{bucket_name}/datasets/wine-quality/'],
            check=True
        )
        print(f"✓ Datos subidos a gs://{bucket_name}/datasets/wine-quality/")


if __name__ == "__main__":
    print("=" * 60)
    print("Preparación de Wine Quality Dataset")
    print("=" * 60)

    wine_data = load_wine_data()
    wine_data = create_quality_binary(wine_data)
    X_train, X_test, y_train, y_test, scaler = prepare_features(wine_data)
    save_processed_data(X_train, X_test, y_train, y_test, scaler)

    print("\n" + "=" * 60)
    print("✓ Preparación completada!")
    print("=" * 60)
```

**Ejecutar preparación**:

```bash
cd ~/mlflow-project
python scripts/prepare_data.py
```

**Output esperado**:
```
============================================================
Preparación de Wine Quality Dataset
============================================================
Total samples: 6497
Red wine: 1599, White wine: 4898

Distribución de calidad:
3      30
4     216
5    2138
6    2836
7    1079
8     193
9       5

Clase positiva (good wine): 1277
Clase negativa (bad wine): 5220
Balance: 19.65%

Train set: 5197 samples
Test set: 1300 samples
Features: ['fixed acidity', 'volatile acidity', 'citric acid', ...]

✓ Datos procesados guardados en data/processed/
✓ Datos subidos a gs://your-project-mlflow-artifacts/datasets/wine-quality/

============================================================
✓ Preparación completada!
============================================================
```

---

------------------------------------------------------------------------

## Parte 4: Experiment Tracking

### 4.1 Script de Training con MLflow Tracking

```python
# Archivo: scripts/train_model.py

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para servidor headless
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile

# Tracking URI apunta a Nginx (puerto 80)
mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:80'))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


def load_data():
    """Carga datos procesados"""
    X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).values.ravel()
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """Genera confusion matrix como archivo temporal"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bad Wine', 'Good Wine'],
                yticklabels=['Bad Wine', 'Good Wine'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    tmpfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(tmpfile.name)
    plt.close()
    return tmpfile.name


def plot_feature_importance(model, feature_names, title='Feature Importance'):
    """Genera plot de feature importance"""
    if not hasattr(model, 'feature_importances_'):
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)),
               [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()

    tmpfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(tmpfile.name)
    plt.close()
    return tmpfile.name


def train_model(model_type='random_forest', hyperparameters=None):
    """
    Entrena modelo con MLflow tracking.

    Args:
        model_type: 'random_forest', 'logistic_regression', o 'svm'
        hyperparameters: dict de hiperparámetros (None usa defaults)
    """
    experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', 'wine-quality-classification')
    mlflow.set_experiment(experiment_name)

    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name=f"{model_type}_training") as run:
        print(f"\n{'=' * 60}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Experiment: {experiment_name}")
        print(f"Model Type: {model_type}")
        print(f"{'=' * 60}\n")

        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        # Seleccionar y configurar modelo
        model_configs = {
            'random_forest': {
                'class': RandomForestClassifier,
                'defaults': {'n_estimators': 100, 'max_depth': 10,
                            'min_samples_split': 5, 'random_state': 42}
            },
            'logistic_regression': {
                'class': LogisticRegression,
                'defaults': {'C': 1.0, 'max_iter': 1000, 'random_state': 42}
            },
            'svm': {
                'class': SVC,
                'defaults': {'C': 1.0, 'kernel': 'rbf',
                            'random_state': 42, 'probability': True}
            }
        }

        if model_type not in model_configs:
            raise ValueError(f"Unknown model_type: {model_type}")

        config = model_configs[model_type]
        params = {**config['defaults'], **(hyperparameters or {})}
        model = config['class'](**params)

        # Log hiperparámetros
        mlflow.log_params(params)

        # Entrenar
        print("Entrenando modelo...")
        model.fit(X_train, y_train)
        print("✓ Entrenamiento completado")

        # Predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]

        # Métricas
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test),
            'test_recall': recall_score(y_test, y_pred_test),
            'test_f1': f1_score(y_test, y_pred_test),
            'test_roc_auc': roc_auc_score(y_test, y_pred_proba_test)
        }
        mlflow.log_metrics(metrics)

        print("\nMétricas:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test,
                                    target_names=['Bad Wine', 'Good Wine']))

        # Artifacts: Confusion Matrix
        cm_file = plot_confusion_matrix(y_test, y_pred_test)
        mlflow.log_artifact(cm_file, "plots")
        os.unlink(cm_file)

        # Artifacts: Feature Importance
        fi_file = plot_feature_importance(model, X_train.columns.tolist())
        if fi_file:
            mlflow.log_artifact(fi_file, "plots")
            os.unlink(fi_file)

        # Log modelo con signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=f"wine-quality-{model_type}"
        )

        print(f"\n✓ Modelo registrado: wine-quality-{model_type}")
        print(f"✓ Run ID: {run.info.run_id}")
        print(f"✓ Artifact URI: {mlflow.get_artifact_uri()}")

        return run.info.run_id, metrics


if __name__ == "__main__":
    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'random_forest'
    run_id, metrics = train_model(model_type=model_type)

    print(f"\n{'=' * 60}")
    print(f"Training completado!")
    print(f"Ver resultados en MLflow UI: {os.environ.get('MLFLOW_TRACKING_URI')}")
    print(f"{'=' * 60}")
```

**Ejecutar training de múltiples modelos**:

```bash
cd ~/mlflow-project

# Random Forest
python scripts/train_model.py random_forest

# Logistic Regression
python scripts/train_model.py logistic_regression

# SVM
python scripts/train_model.py svm
```

### 4.2 Comparar Experimentos en MLflow UI

Abrir en el navegador:

```bash
echo "MLflow UI: http://$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip -H 'Metadata-Flavor: Google')"
```

**En el MLflow UI** (puerto 80):

1. Ver experiment "wine-quality-classification"
2. Comparar runs de diferentes modelos
3. Ordenar por `test_accuracy` o `test_f1`
4. Ver confusion matrices y feature importance en la pestaña Artifacts → plots/
5. Comparar parámetros side-by-side

---

## Parte 5: Model Registry

### Contexto: ¿Qué hay registrado tras la Parte 4?

Al ejecutar el training de la Parte 4, el script `train_model.py` usa `registered_model_name=f"wine-quality-{model_type}"` en `mlflow.sklearn.log_model()`. Esto significa que **cada tipo de modelo se registra automáticamente** como un modelo independiente en el Registry:

```
Model Registry (tras Parte 4):
├── wine-quality-random_forest     → Version 1 (del run de Random Forest)
├── wine-quality-logistic_regression → Version 1 (del run de Logistic Regression)
└── wine-quality-svm               → Version 1 (del run de SVM)
```

Si ejecutas el mismo tipo de modelo varias veces (por ejemplo, Random Forest con diferentes hiperparámetros), MLflow **no crea un nuevo modelo** sino que **añade una nueva versión** al existente:

```
wine-quality-random_forest:
├── Version 1 → n_estimators=100, max_depth=10 (primer training)
├── Version 2 → n_estimators=200, max_depth=15 (segundo training)
└── Version 3 → n_estimators=50, max_depth=20  (tercer training)
```

Cada versión mantiene la trazabilidad completa: qué run la generó, con qué hiperparámetros, qué métricas obtuvo, y dónde están los artifacts en GCS.

### Stages del Model Registry

MLflow organiza las versiones de cada modelo en **stages** (estados) que representan el ciclo de vida:

```
                    ┌───────────┐
   Registro ──────▶ │   None    │  ← Estado inicial al registrar
                    └─────┬─────┘
                          │ transition_model_version_stage()
                          ▼
                    ┌───────────┐
                    │  Staging  │  ← Validación / testing pre-producción
                    └─────┬─────┘
                          │ transition_model_version_stage()
                          ▼
                    ┌───────────┐
                    │Production │  ← Modelo activo sirviendo predicciones
                    └─────┬─────┘
                          │ (al promover nueva versión)
                          ▼
                    ┌───────────┐
                    │ Archived  │  ← Versiones anteriores retiradas
                    └───────────┘
```

**Importante**: `archive_existing_versions=True` en la transición asegura que al promover una versión a Production, la versión anterior se mueve automáticamente a Archived. Solo puede haber **una versión en Production** por modelo registrado.

### 5.1 Registrar Mejor Modelo

El script `register_model.py` busca el mejor run **entre todos los modelos** del experimento (Random Forest, Logistic Regression, SVM), lo registra con un nombre especial (`-best`) y lo transiciona a Staging:

```python
# Archivo: scripts/register_model.py

import mlflow
from mlflow.tracking import MlflowClient
import os

mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:80'))


def register_best_model(experiment_name, metric='test_f1'):
    """
    Busca el mejor run del experimento (independientemente del tipo de modelo)
    y lo registra en el Model Registry con nombre 'wine-quality-{tipo}-best'.

    Esto crea un modelo ADICIONAL al que ya se registró en train_model.py.
    Por ejemplo, si el mejor modelo es un Random Forest, el Registry tendrá:
      - wine-quality-random_forest       → todas las versiones de RF
      - wine-quality-random_forest-best  → solo la mejor versión global
    """
    client = MlflowClient()

    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )

    if len(runs) == 0:
        print("No runs encontrados")
        return None

    best_run = runs.iloc[0]
    run_id = best_run['run_id']
    model_type = best_run['params.model_type']

    print(f"Registrando modelo desde run {run_id}")
    print(f"Model type: {model_type}")
    print(f"{metric}: {best_run[f'metrics.{metric}']:.4f}")

    model_uri = f"runs:/{run_id}/model"
    registered_name = f"wine-quality-{model_type}-best"

    model_details = mlflow.register_model(model_uri=model_uri, name=registered_name)

    client.update_model_version(
        name=registered_name,
        version=model_details.version,
        description=f"Best {model_type} model - {metric}: {best_run[f'metrics.{metric}']:.4f}"
    )

    client.set_model_version_tag(
        name=registered_name, version=model_details.version,
        key="metric_used", value=metric
    )
    client.set_model_version_tag(
        name=registered_name, version=model_details.version,
        key="model_type", value=model_type
    )

    print(f"\n✓ Modelo registrado: {registered_name}")
    print(f"✓ Version: {model_details.version}")

    return model_details


def transition_model_stage(model_name, version, stage):
    """
    Transiciona modelo a stage (Staging/Production/Archived).

    Con archive_existing_versions=True, al mover a Production se archiva
    automáticamente cualquier versión que estuviera previamente en Production.
    """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name, version=version, stage=stage,
        archive_existing_versions=True
    )
    print(f"✓ Modelo {model_name} v{version} → {stage}")


def list_registered_models():
    """
    Lista TODOS los modelos registrados y sus versiones.

    Tras ejecutar train_model.py (3 modelos) + register_model.py, veremos:
      - wine-quality-random_forest          (de train_model.py)
      - wine-quality-logistic_regression    (de train_model.py)
      - wine-quality-svm                    (de train_model.py)
      - wine-quality-{best_type}-best       (de register_model.py)
    """
    client = MlflowClient()

    print(f"\n{'=' * 60}")
    print("Modelos Registrados")
    print(f"{'=' * 60}")

    for rm in client.search_registered_models():
        print(f"\nModelo: {rm.name}")
        for mv in rm.latest_versions:
            print(f"  - Version {mv.version}: {mv.current_stage} (run: {mv.run_id[:8]}...)")


if __name__ == "__main__":
    # 1. Buscar y registrar el mejor modelo global
    model_details = register_best_model("wine-quality-classification", metric='test_f1')

    if model_details:
        # 2. Mover a Staging para validación
        transition_model_stage(model_details.name, model_details.version, "Staging")
        # 3. Mostrar estado completo del Registry
        list_registered_models()
```

**Ejecutar**:
```bash
python scripts/register_model.py
```

**Output esperado** (ejemplo donde Random Forest fue el mejor):
```
Registrando modelo desde run a1b2c3d4e5f6...
Model type: random_forest
test_f1: 0.6234

✓ Modelo registrado: wine-quality-random_forest-best
✓ Version: 1
✓ Modelo wine-quality-random_forest-best v1 → Staging

============================================================
Modelos Registrados
============================================================

Modelo: wine-quality-random_forest
  - Version 1: None (run: a1b2c3d4...)

Modelo: wine-quality-logistic_regression
  - Version 1: None (run: e5f6g7h8...)

Modelo: wine-quality-svm
  - Version 1: None (run: i9j0k1l2...)

Modelo: wine-quality-random_forest-best
  - Version 1: Staging (run: a1b2c3d4...)
```

**Nota**: Observa que `wine-quality-random_forest` (Version 1, stage None) y `wine-quality-random_forest-best` (Version 1, stage Staging) **apuntan al mismo run**. Son dos entradas en el Registry, pero referencian el mismo modelo entrenado y los mismos artifacts en GCS. La diferencia es semántica: el `-best` identifica al ganador de la competición entre modelos.

### 5.2 Gestión de Versiones y Promoción a Production

Este script consolida las operaciones de consulta y promoción de modelos. Muestra el estado completo del Registry, compara métricas entre versiones, y permite promover un modelo de Staging a Production de forma interactiva.

Antes de poder levantar el serving (Parte 6), **es necesario tener un modelo en Production**. Este script facilita ese paso.

```python
# Archivo: scripts/manage_model_versions.py

from mlflow.tracking import MlflowClient
import mlflow
import os

mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:80'))


def show_registry_status():
    """
    Muestra el estado completo del Model Registry:
    - Todos los modelos registrados
    - Todas sus versiones con stage y métricas principales
    - Identifica cuál está en Production (si hay alguno)
    """
    client = MlflowClient()
    models = list(client.search_registered_models())

    if not models:
        print("⚠ No hay modelos registrados. Ejecuta primero train_model.py y register_model.py")
        return None

    print(f"\n{'=' * 70}")
    print("Estado del Model Registry")
    print(f"{'=' * 70}")

    production_model = None

    for rm in models:
        print(f"\n📦 Modelo: {rm.name}")

        for mv in rm.latest_versions:
            # Obtener métricas del run asociado
            run = client.get_run(mv.run_id)
            metrics = run.data.metrics

            stage_icon = {
                'Production': '🟢', 'Staging': '🟡',
                'Archived': '⚫', 'None': '⚪'
            }.get(mv.current_stage, '⚪')

            print(f"  {stage_icon} Version {mv.version} [{mv.current_stage}]")
            print(f"      Run: {mv.run_id[:12]}...")

            # Mostrar métricas clave
            for key in ['test_f1', 'test_accuracy', 'test_roc_auc']:
                if key in metrics:
                    print(f"      {key}: {metrics[key]:.4f}")

            if mv.current_stage == 'Production':
                production_model = (rm.name, mv.version)

    print(f"\n{'─' * 70}")
    if production_model:
        print(f"✅ Modelo en Production: {production_model[0]} v{production_model[1]}")
    else:
        print("⚠  No hay ningún modelo en Production")
    print(f"{'─' * 70}")

    return production_model


def compare_model_versions(model_name, version_1, version_2):
    """
    Compara métricas lado a lado de dos versiones del mismo modelo.
    Útil para decidir si promover una nueva versión.
    """
    client = MlflowClient()

    print(f"\n{'=' * 60}")
    print(f"Comparación: {model_name}")
    print(f"{'=' * 60}")

    for ver in [version_1, version_2]:
        mv = client.get_model_version(model_name, ver)
        run = client.get_run(mv.run_id)
        print(f"\n  Version {ver} ({mv.current_stage}):")
        for key, value in sorted(run.data.metrics.items()):
            print(f"    {key}: {value:.4f}")


def promote_to_production(model_name, version):
    """
    Promueve una versión a Production.
    Archiva automáticamente cualquier versión previamente en Production.
    """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name, version=version,
        stage='Production', archive_existing_versions=True
    )
    print(f"\n✅ {model_name} v{version} → Production")
    print("   (versiones anteriores en Production han sido archivadas)")


def load_model_from_registry(model_name, stage='Production'):
    """Carga modelo desde registry para verificar que funciona"""
    model_uri = f"models:/{model_name}/{stage}"
    print(f"\nCargando modelo: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✓ Modelo cargado correctamente: {type(model).__name__}")
    return model


if __name__ == "__main__":
    # 1. Mostrar estado actual del Registry
    production_model = show_registry_status()

    # 2. Buscar modelos en Staging (candidatos a promoción)
    client = MlflowClient()
    staging_candidates = []

    for rm in client.search_registered_models():
        for mv in rm.latest_versions:
            if mv.current_stage == 'Staging':
                staging_candidates.append((rm.name, mv.version))

    # 3. Si hay candidatos en Staging, preguntar si promover
    if staging_candidates:
        print(f"\n🟡 Modelos en Staging (candidatos a Production):")
        for i, (name, ver) in enumerate(staging_candidates):
            print(f"  [{i + 1}] {name} v{ver}")

        print(f"\n¿Deseas promover algún modelo a Production?")
        print(f"  Introduce el número (1-{len(staging_candidates)}) o 'n' para cancelar: ", end='')
        choice = input().strip().lower()

        if choice != 'n' and choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(staging_candidates):
                name, ver = staging_candidates[idx]

                # Si ya hay modelo en Production, mostrar comparación
                if production_model:
                    prod_name, prod_ver = production_model
                    if prod_name == name:
                        print(f"\nComparando versión actual en Production vs candidato:")
                        compare_model_versions(name, prod_ver, ver)

                    confirm = input(f"\n¿Confirmas promoción de {name} v{ver} a Production? (s/n): ").strip().lower()
                else:
                    confirm = input(f"\n¿Confirmas promoción de {name} v{ver} a Production? (s/n): ").strip().lower()

                if confirm == 's':
                    promote_to_production(name, ver)
                    # Verificar que el modelo se carga correctamente
                    load_model_from_registry(name, stage='Production')
                else:
                    print("Cancelado.")
            else:
                print("Selección no válida.")
        else:
            print("Sin cambios.")

    elif not production_model:
        print("\n⚠ No hay modelos en Staging ni en Production.")
        print("  Ejecuta primero: python scripts/register_model.py")
    else:
        print("\nNo hay modelos en Staging pendientes de promoción.")
```

**Ejecutar**:
```bash
python scripts/manage_model_versions.py
```

**Output esperado** (ejemplo interactivo):
```
======================================================================
Estado del Model Registry
======================================================================

📦 Modelo: wine-quality-random_forest
  ⚪ Version 1 [None]
      Run: a1b2c3d4e5f6...
      test_f1: 0.6234
      test_accuracy: 0.8723
      test_roc_auc: 0.8456

📦 Modelo: wine-quality-logistic_regression
  ⚪ Version 1 [None]
      Run: e5f6g7h8i9j0...
      test_f1: 0.5891
      test_accuracy: 0.8612
      test_roc_auc: 0.8201

📦 Modelo: wine-quality-svm
  ⚪ Version 1 [None]
      Run: k1l2m3n4o5p6...
      test_f1: 0.6012
      test_accuracy: 0.8689
      test_roc_auc: 0.8334

📦 Modelo: wine-quality-random_forest-best
  🟡 Version 1 [Staging]
      Run: a1b2c3d4e5f6...
      test_f1: 0.6234
      test_accuracy: 0.8723
      test_roc_auc: 0.8456

──────────────────────────────────────────────────────────────────────
⚠  No hay ningún modelo en Production
──────────────────────────────────────────────────────────────────────

🟡 Modelos en Staging (candidatos a Production):
  [1] wine-quality-random_forest-best v1

¿Deseas promover algún modelo a Production?
  Introduce el número (1-1) o 'n' para cancelar: 1

¿Confirmas promoción de wine-quality-random_forest-best v1 a Production? (s/n): s

✅ wine-quality-random_forest-best v1 → Production
   (versiones anteriores en Production han sido archivadas)

Cargando modelo: models:/wine-quality-random_forest-best/Production
✓ Modelo cargado correctamente: RandomForestClassifier
```

**Verificar en MLflow UI**: En la pestaña "Models" del UI (http://<VM_IP>) se pueden ver todos los modelos registrados, sus versiones y stages de forma visual.

---

## Parte 6: Model Serving

### 6.1 Preparar el Serving

El contenedor de serving está configurado en `docker-compose.yml` con `profiles: [serving]`, lo que significa que **no arranca con `docker-compose up -d`** — hay que levantarlo explícitamente. Esto es intencional: el serving consume ~400-768 MB de RAM y solo tiene sentido cuando hay un modelo en Production.

El contenedor busca el modelo usando la URI `models:/${MODEL_NAME}/Production`, donde `MODEL_NAME` está definido como `wine-quality-random_forest-best` en el docker-compose. Este nombre coincide con el que `register_model.py` (Parte 5.1) asigna al mejor modelo. Si tu mejor modelo resultó ser otro tipo (por ejemplo logistic_regression), el nombre sería `wine-quality-logistic_regression-best` y habría que actualizarlo en el docker-compose.

### 6.2 Ajustar el Nombre del Modelo y Levantar

```bash
# ─── Verificar qué modelo está en Production ───
python -c "
from mlflow.tracking import MlflowClient
import mlflow, os
mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:80'))
client = MlflowClient()
for rm in client.search_registered_models():
    for mv in rm.latest_versions:
        if mv.current_stage == 'Production':
            print(f'Production: {rm.name} v{mv.version}')
"

# ─── Si el nombre no coincide con wine-quality-pipeline, actualizar ───
# Ejemplo: si el modelo en Production es wine-quality-random_forest-best
# Editar docker-compose.yml y cambiar MODEL_NAME:
#   MODEL_NAME: wine-quality-random_forest-best

# ─── Levantar serving ───
docker-compose --profile serving up -d mlflow-serving

# Verificar que arranca
docker-compose ps

# Esperar a que cargue el modelo (~30-60 segundos)
sleep 30

# Ver logs (confirmar que cargó bien)
docker-compose logs --tail=20 mlflow-serving

# Test de salud vía Nginx
curl http://localhost/health/serving
# Output: {"status": "OK"}
```

**Si el contenedor falla al arrancar**, revisar los logs:
```bash
docker-compose logs mlflow-serving
```

Las causas más comunes son:
- **No hay modelo en Production**: Ejecutar primero el paso 5.3
- **MODEL_NAME no coincide**: El nombre en docker-compose.yml debe coincidir exactamente con el nombre del modelo registrado que tiene una versión en Production
- **Credenciales GCS**: El contenedor de serving también necesita acceder a GCS para descargar los artifacts del modelo

### 6.3 Test de Predicción

Una vez que el serving está corriendo, se puede enviar requests de predicción a través de Nginx:

```bash
# Predicción de ejemplo (11 features fisicoquímicas + wine_type)
curl -X POST http://localhost/api/predict \
    -H "Content-Type: application/json" \
    -d '{
        "dataframe_split": {
            "columns": ["fixed acidity", "volatile acidity", "citric acid",
                        "residual sugar", "chlorides", "free sulfur dioxide",
                        "total sulfur dioxide", "density", "pH",
                        "sulphates", "alcohol", "wine_type"],
            "data": [[0.5, -0.3, 0.1, -0.2, 0.0, 0.3, -0.1, 0.2, -0.4, 0.1, 0.8, 1]]
        }
    }'
# Output: {"predictions": [0]}  (0=bad wine, 1=good wine)
```

**Nota**: Los valores de entrada deben estar **escalados** (StandardScaler), ya que el modelo fue entrenado con datos normalizados. En un entorno de producción real, el escalado se integraría en un pipeline de scikit-learn o como un paso de preprocesamiento en el servidor.

---

## Parte 7: Pipeline Automatizado

### Contexto: ¿Por qué un pipeline automatizado?

Hasta ahora hemos ejecutado cada paso manualmente: preparar datos, entrenar modelos uno a uno, registrar el mejor, promover a Staging/Production, y levantar el serving. En un entorno MLOps real, todo esto debería poder ejecutarse **con un solo comando**, incluyendo:

- Hyperparameter tuning automático (no solo defaults)
- Quality gates: el modelo solo avanza si supera umbrales mínimos
- Comparación con el modelo actual en Production antes de reemplazarlo
- Reinicio automático del contenedor de serving si hay nuevo modelo

### Flujo del Pipeline

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  [1] Cargar  │───▶│  [2] Hyper-param │───▶│  [3] Train +     │
│     datos    │    │     tuning       │    │     evaluate     │
└─────────────┘    │  GridSearchCV    │    │  MLflow tracking │
                   └──────────────────┘    └────────┬─────────┘
                                                    │
                                                    ▼
                                           ┌──────────────────┐
                                  ¿F1 ≥ 0.50?│  [4] Registrar   │
                                  NO → STOP  │     modelo       │
                                  SÍ ↓       └────────┬─────────┘
                                                    │
                                                    ▼
                                           ┌──────────────────┐
                                           │  [5] Promover a  │
                                           │     Staging      │
                                           └────────┬─────────┘
                                                    │
                                                    ▼
                                           ┌──────────────────┐
                                  ¿F1 ≥ 0.55?│  [6] Validar y   │
                                  Y > prod?  │  promover a Prod │
                                  NO → STOP  └────────┬─────────┘
                                  SÍ ↓                │
                                                    ▼
                                           ┌──────────────────┐
                                           │  [7] Restart     │
                                           │     serving      │
                                           └──────────────────┘
```

**Quality gates** (umbrales): El pipeline tiene dos niveles de exigencia:
- **`min_f1_register=0.50`**: Umbral mínimo para que el modelo se registre en el Registry. Por debajo, el modelo se descarta. Es un umbral bajo intencionalmente — en este dataset desbalanceado (80% clase negativa), un F1 de 0.50 ya indica que el modelo hace algo útil.
- **`min_f1_production=0.55`**: Umbral para promover a Production. Además, el nuevo modelo debe **superar al actual** en Production (si existe). Esto evita reemplazar un modelo bueno por uno peor.

Estos valores son conservadores para el Wine Quality dataset. En un proyecto real, se ajustarían según los requisitos del negocio.

### 7.1 Pipeline Completo End-to-End

```python
# Archivo: scripts/mlops_pipeline.py

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os
import subprocess
from datetime import datetime

mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:80'))
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

# ─── Nombre del modelo en el Registry ───
# Debe coincidir con MODEL_NAME en docker-compose.yml (sección serving)
# y con el nombre usado en register_model.py (Parte 5.1)
REGISTERED_MODEL_NAME = "wine-quality-random_forest-best"


class MLOpsPipeline:
    """
    Pipeline MLOps automatizado end-to-end.

    Ejecuta secuencialmente: carga de datos → hyperparameter tuning →
    training con tracking → registro condicional → promoción a Staging →
    validación y promoción a Production → restart del serving.

    El nombre del modelo registrado (REGISTERED_MODEL_NAME) es el mismo
    que se usó en las Partes 5 y 6. Así, las nuevas versiones se añaden
    al mismo modelo y el serving container puede cargarlas sin cambiar
    su configuración.
    """

    def __init__(self, experiment_name="wine-quality-classification"):
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        self.model_name = REGISTERED_MODEL_NAME
        mlflow.set_experiment(experiment_name)

    def load_data(self):
        """Step 1: Cargar datos procesados (generados en Parte 3)"""
        print("\n[1/7] Cargando datos...")
        X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
        y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).values.ravel()
        y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).values.ravel()

        print(f"  ✓ Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        print(f"  ✓ Features: {X_train.shape[1]}")
        print(f"  ✓ Balance clase positiva: {y_train.mean():.2%} train, {y_test.mean():.2%} test")
        return X_train, X_test, y_train, y_test

    def hyperparameter_tuning(self, X_train, y_train):
        """
        Step 2: Hyperparameter tuning con GridSearchCV.

        Usa cross-validation 5-fold con F1 como métrica de scoring.
        GridSearchCV prueba todas las combinaciones de la grid (3x3x3 = 27)
        y devuelve el mejor estimador ya entrenado.

        n_jobs=-1 usa todos los cores disponibles (2 en e2-standard-2).
        """
        print("\n[2/7] Hyperparameter tuning (esto tarda ~1-2 min)...")

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }

        total_fits = len(param_grid['n_estimators']) * \
                     len(param_grid['max_depth']) * \
                     len(param_grid['min_samples_split']) * 5  # 5 folds
        print(f"  Probando {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split'])} combinaciones × 5 folds = {total_fits} fits")

        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)

        print(f"\n  ✓ Best params: {grid_search.best_params_}")
        print(f"  ✓ Best CV F1: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def train_and_evaluate(self, model, X_train, X_test, y_train, y_test, params):
        """
        Step 3: Re-entrenar con mejores params y evaluar en test set.

        Aunque GridSearchCV ya entrenó el modelo, aquí lo hacemos dentro
        de un mlflow.start_run() para que quede registrado con todos los
        parámetros, métricas y artifacts.

        El run_name incluye timestamp para identificar fácilmente cada
        ejecución del pipeline en el MLflow UI.
        """
        print("\n[3/7] Training final y evaluación con MLflow tracking...")

        with mlflow.start_run(run_name=f"pipeline_{datetime.now():%Y%m%d_%H%M%S}") as run:
            # Log parámetros
            mlflow.log_params(params)
            mlflow.log_param("pipeline_type", "automated")
            mlflow.log_param("model_type", "random_forest")
            mlflow.log_param("tuning_method", "GridSearchCV_5fold")
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))

            # Entrenar (rápido, ya está fitted pero re-fit para consistencia)
            model.fit(X_train, y_train)

            # Evaluar
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                'test_accuracy': accuracy_score(y_test, y_pred),
                'test_f1': f1_score(y_test, y_pred),
                'test_roc_auc': roc_auc_score(y_test, y_proba)
            }
            mlflow.log_metrics(metrics)

            print(f"  ✓ Métricas en test set:")
            for name, val in metrics.items():
                print(f"    {name}: {val:.4f}")

            # Log modelo con signature (para validación de inputs en serving)
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                model, artifact_path="model", signature=signature
            )

            print(f"  ✓ Run ID: {run.info.run_id}")
            print(f"  ✓ Artifacts en: {mlflow.get_artifact_uri()}")

            return run.info.run_id, metrics

    def register_model(self, run_id, metrics, min_f1_register=0.50):
        """
        Step 4: Registrar modelo en el Registry (quality gate #1).

        Solo registra si F1 ≥ min_f1_register. Esto evita llenar el Registry
        de versiones que no sirven.

        El modelo se registra con el MISMO nombre (REGISTERED_MODEL_NAME)
        usado en las Partes 5 y 6. MLflow crea una nueva versión automáticamente:
          wine-quality-random_forest-best v1 → (Parte 5)
          wine-quality-random_forest-best v2 → (Parte 5, segundo training)
          wine-quality-random_forest-best v3 → (este pipeline)
        """
        print(f"\n[4/7] Registro en Model Registry (threshold F1 ≥ {min_f1_register})...")

        if metrics['test_f1'] < min_f1_register:
            print(f"  ✗ F1 ({metrics['test_f1']:.4f}) < threshold ({min_f1_register})")
            print(f"  ✗ Modelo NO registrado — no supera quality gate")
            return None

        model_details = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=self.model_name
        )

        self.client.update_model_version(
            name=self.model_name, version=model_details.version,
            description=f"Pipeline auto {datetime.now():%Y-%m-%d %H:%M} — "
                        f"F1: {metrics['test_f1']:.4f}, "
                        f"AUC: {metrics['test_roc_auc']:.4f}"
        )

        print(f"  ✓ Registrado: {self.model_name} v{model_details.version}")
        return model_details

    def promote_to_staging(self, model_details):
        """
        Step 5: Promover nueva versión a Staging.

        Staging es un paso intermedio de validación. En un entorno real,
        aquí se ejecutarían tests de integración, validación de datos,
        smoke tests, etc. antes de promover a Production.
        """
        print("\n[5/7] Promoviendo a Staging...")

        if not model_details:
            print("  ⚠ No hay modelo para promover (no pasó quality gate)")
            return False

        self.client.transition_model_version_stage(
            name=self.model_name, version=model_details.version, stage="Staging"
        )
        print(f"  ✓ {self.model_name} v{model_details.version} → Staging")
        return True

    def validate_and_promote_production(self, model_details, metrics, min_f1_production=0.55):
        """
        Step 6: Validar y promover a Production (quality gate #2).

        Dos condiciones para promover:
        1. F1 ≥ min_f1_production (umbral absoluto)
        2. F1 > F1 del modelo actual en Production (mejora relativa)

        Si no hay modelo en Production (primera vez), solo aplica condición 1.
        archive_existing_versions=True archiva la versión anterior automáticamente.
        """
        print(f"\n[6/7] Validación para Production (threshold F1 ≥ {min_f1_production})...")

        if not model_details:
            print("  ⚠ No hay modelo para promover")
            return False

        # Quality gate #2: umbral absoluto
        if metrics['test_f1'] < min_f1_production:
            print(f"  ✗ F1 ({metrics['test_f1']:.4f}) < production threshold ({min_f1_production})")
            print(f"  ✗ Se queda en Staging, no se promueve a Production")
            return False

        # Comparar con modelo actual en Production (si existe)
        prod_versions = self.client.get_latest_versions(
            self.model_name, stages=["Production"]
        )

        if prod_versions:
            prod_run = self.client.get_run(prod_versions[0].run_id)
            prod_f1 = prod_run.data.metrics.get('test_f1', 0)
            prod_version = prod_versions[0].version

            print(f"  Comparando con Production actual:")
            print(f"    Actual  (v{prod_version}): F1 = {prod_f1:.4f}")
            print(f"    Nuevo   (v{model_details.version}): F1 = {metrics['test_f1']:.4f}")

            if metrics['test_f1'] <= prod_f1:
                print(f"  ✗ Nuevo modelo no mejora al actual — se queda en Staging")
                return False

            improvement = metrics['test_f1'] - prod_f1
            print(f"  ✓ Mejora de +{improvement:.4f} en F1")
        else:
            print(f"  No hay modelo en Production — primera promoción")

        # Promover
        self.client.transition_model_version_stage(
            name=self.model_name, version=model_details.version,
            stage="Production", archive_existing_versions=True
        )
        print(f"  ✓ {self.model_name} v{model_details.version} → Production")

        return True

    def restart_serving(self):
        """
        Step 7: Reiniciar contenedor de serving para cargar el nuevo modelo.

        El contenedor de serving carga el modelo al arrancar usando
        models:/{MODEL_NAME}/Production. Al reiniciarlo, descarga la nueva
        versión del modelo desde GCS.

        Nota: Se usa 'docker-compose' (con guión) para compatibilidad con
        instalaciones vía apt. Si tienes el Docker Compose plugin (v2),
        'docker compose' (sin guión) también funciona.
        """
        print("\n[7/7] Reiniciando serving container...")
        project_dir = os.path.expanduser('~/mlflow-project')

        # Intentar primero docker-compose (standalone), luego docker compose (plugin)
        commands = [
            ['docker-compose', '--profile', 'serving',
             'up', '-d', '--force-recreate', 'mlflow-serving'],
            ['docker', 'compose', '--profile', 'serving',
             'up', '-d', '--force-recreate', 'mlflow-serving'],
        ]

        for cmd in commands:
            try:
                subprocess.run(
                    cmd, cwd=project_dir,
                    check=True, capture_output=True
                )
                print(f"  ✓ Serving container reiniciado (usando {cmd[0]})")
                print("  ⏳ Esperar ~30-60 seg para que cargue el modelo")
                print(f"  📡 Test: curl http://localhost/health/serving")
                return
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

        print("  ✗ No se pudo reiniciar el serving automáticamente")
        print("  Ejecutar manualmente:")
        print("    docker-compose --profile serving up -d --force-recreate mlflow-serving")

    def show_summary(self, run_id, metrics, model_details, promoted):
        """Resumen final del pipeline"""
        print(f"\n{'═' * 60}")
        print("RESUMEN DEL PIPELINE")
        print(f"{'═' * 60}")
        print(f"  Run ID:           {run_id}")
        print(f"  Modelo:           {self.model_name}")

        if model_details:
            print(f"  Versión:          v{model_details.version}")
        else:
            print(f"  Versión:          (no registrado)")

        print(f"  F1 Score:         {metrics['test_f1']:.4f}")
        print(f"  Accuracy:         {metrics['test_accuracy']:.4f}")
        print(f"  ROC AUC:          {metrics['test_roc_auc']:.4f}")

        if promoted:
            print(f"  Estado:           ✅ PROMOVIDO A PRODUCTION")
            print(f"  Serving:          Reiniciado con nueva versión")
        elif model_details:
            print(f"  Estado:           🟡 EN STAGING (no superó validación para Production)")
        else:
            print(f"  Estado:           ⚫ NO REGISTRADO (no superó quality gate)")

        print(f"{'═' * 60}")
        print(f"  MLflow UI: {os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:80')}")
        print(f"{'═' * 60}")

    def run(self):
        """Ejecutar pipeline completo"""
        print("\n" + "=" * 60)
        print("MLOps Pipeline Automatizado")
        print(f"Modelo: {self.model_name}")
        print(f"Timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 60)

        # Step 1: Datos
        X_train, X_test, y_train, y_test = self.load_data()

        # Step 2: Tuning
        model, best_params = self.hyperparameter_tuning(X_train, y_train)

        # Step 3: Train + evaluate
        run_id, metrics = self.train_and_evaluate(
            model, X_train, X_test, y_train, y_test, best_params
        )

        # Step 4: Register (quality gate #1)
        model_details = self.register_model(run_id, metrics)

        # Step 5: Staging
        self.promote_to_staging(model_details)

        # Step 6: Production (quality gate #2)
        promoted = self.validate_and_promote_production(model_details, metrics)

        # Step 7: Restart serving (solo si se promovió)
        if promoted:
            self.restart_serving()

        # Resumen
        self.show_summary(run_id, metrics, model_details, promoted)

        return {
            'run_id': run_id,
            'metrics': metrics,
            'model_details': model_details,
            'promoted_to_production': promoted
        }


if __name__ == "__main__":
    pipeline = MLOpsPipeline()
    results = pipeline.run()
```

### 7.2 Ejecutar el Pipeline

```bash
cd ~/mlflow-project
python scripts/mlops_pipeline.py
```

**Output esperado** (ejemplo donde el pipeline completa todos los pasos):

```
============================================================
MLOps Pipeline Automatizado
Modelo: wine-quality-random_forest-best
Timestamp: 2026-02-22 15:30:45
============================================================

[1/7] Cargando datos...
  ✓ Train: 5197 samples, Test: 1300 samples
  ✓ Features: 12
  ✓ Balance clase positiva: 19.63% train, 19.69% test

[2/7] Hyperparameter tuning (esto tarda ~1-2 min)...
  Probando 27 combinaciones × 5 folds = 135 fits
Fitting 5 folds for each of 27 candidates, totalling 135 fits

  ✓ Best params: {'max_depth': 15, 'min_samples_split': 5, 'n_estimators': 200}
  ✓ Best CV F1: 0.6312

[3/7] Training final y evaluación con MLflow tracking...
  ✓ Métricas en test set:
    test_accuracy: 0.8785
    test_f1: 0.6389
    test_roc_auc: 0.8523
  ✓ Run ID: 7f8a9b2c3d4e...
  ✓ Artifacts en: gs://banded-pad-481109-q1-mlflow-artifacts/mlflow-artifacts/...

[4/7] Registro en Model Registry (threshold F1 ≥ 0.50)...
  ✓ Registrado: wine-quality-random_forest-best v3

[5/7] Promoviendo a Staging...
  ✓ wine-quality-random_forest-best v3 → Staging

[6/7] Validación para Production (threshold F1 ≥ 0.55)...
  Comparando con Production actual:
    Actual  (v2): F1 = 0.6234
    Nuevo   (v3): F1 = 0.6389
  ✓ Mejora de +0.0155 en F1
  ✓ wine-quality-random_forest-best v3 → Production

[7/7] Reiniciando serving container...
  ✓ Serving container reiniciado
  ⏳ Esperar ~30-60 seg para que cargue el modelo
  📡 Test: curl http://localhost/health/serving

════════════════════════════════════════════════════════════
RESUMEN DEL PIPELINE
════════════════════════════════════════════════════════════
  Run ID:           7f8a9b2c3d4e...
  Modelo:           wine-quality-random_forest-best
  Versión:          v3
  F1 Score:         0.6389
  Accuracy:         0.8785
  ROC AUC:          0.8523
  Estado:           ✅ PROMOVIDO A PRODUCTION
  Serving:          Reiniciado con nueva versión
════════════════════════════════════════════════════════════
  MLflow UI: http://localhost:80
════════════════════════════════════════════════════════════
```

### 7.3 Verificar tras el Pipeline

```bash
# 1. Verificar que el serving cargó el nuevo modelo
sleep 30
curl http://localhost/health/serving

# 2. Test de predicción con el nuevo modelo
curl -X POST http://localhost/api/predict \
    -H "Content-Type: application/json" \
    -d '{
        "dataframe_split": {
            "columns": ["fixed acidity", "volatile acidity", "citric acid",
                        "residual sugar", "chlorides", "free sulfur dioxide",
                        "total sulfur dioxide", "density", "pH",
                        "sulphates", "alcohol", "wine_type"],
            "data": [[0.5, -0.3, 0.1, -0.2, 0.0, 0.3, -0.1, 0.2, -0.4, 0.1, 0.8, 1]]
        }
    }'

# 3. Ver estado del Registry
python -c "
from mlflow.tracking import MlflowClient
import mlflow, os
mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:80'))
client = MlflowClient()
for rm in client.search_registered_models():
    print(f'\nModelo: {rm.name}')
    for mv in rm.latest_versions:
        stage_icon = {'Production': '🟢', 'Staging': '🟡', 'Archived': '⚫'}.get(mv.current_stage, '⚪')
        print(f'  {stage_icon} v{mv.version} [{mv.current_stage}]')
"

# 4. Ver todos los runs del pipeline en MLflow UI
echo "MLflow UI: http://$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip -H 'Metadata-Flavor: Google')"
```

### 7.4 Escenarios del Pipeline

El pipeline se comporta diferente según la calidad del modelo entrenado:

| Escenario | F1 obtenido | Resultado |
|-----------|-------------|-----------|
| **Modelo malo** | F1 < 0.50 | No se registra. Run queda en MLflow para análisis pero no entra al Registry |
| **Modelo aceptable** | 0.50 ≤ F1 < 0.55 | Se registra y promueve a Staging. No llega a Production |
| **Modelo bueno (primera vez)** | F1 ≥ 0.55, sin Production previa | Se registra → Staging → Production. Serving se levanta |
| **Modelo bueno (mejora)** | F1 ≥ 0.55 y F1 > actual Production | Se registra → Staging → Production. Serving se reinicia |
| **Modelo bueno (no mejora)** | F1 ≥ 0.55 pero F1 ≤ actual Production | Se registra → Staging. Production no cambia |

Cada ejecución del pipeline crea un nuevo run en el experimento `wine-quality-classification`, visible en el MLflow UI junto a los runs manuales de la Parte 4. Esto permite comparar los resultados del pipeline automatizado con los entrenamientos manuales anteriores.
