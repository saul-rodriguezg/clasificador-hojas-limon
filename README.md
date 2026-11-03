# Clasificador de hojas de limón — ResNet50 + FastAPI + Docker

Aplicación web académica para clasificar fotografías de hojas de limón en dos categorías: `diseased` (hoja enferma) y `healthy` (hoja sana). El backend expone una API REST con FastAPI y sirve el frontend desde el mismo contenedor.

## Estado de la solución

- Modelo final `models/ResNet50.keras` incorporado al paquete ZIP.
- Estructura del modelo inspeccionada: entrada `(None, 224, 224, 3)`, salida de dos neuronas `softmax` y preprocesador KerasHub serializado dentro del `.keras`.
- API y frontend implementados.
- Dockerfile y configuración de despliegue preparados.
- Pruebas de contrato, estructura del artefacto y validaciones HTTP incluidas.
- **Pendiente de entorno:** ejecutar la carga/inferencia real con TensorFlow y construir/ejecutar Docker. Este entorno de preparación no dispone de TensorFlow ni del motor Docker; las instrucciones reproducibles están documentadas abajo.

## Arquitectura

```text
Navegador
   │ fotografía (multipart/form-data)
   ▼
Frontend HTML / JavaScript
   │ POST /predict
   ▼
FastAPI ───── GET /health
   │
   ▼
ResNet50.keras
   ├─ ResNetImageClassifierPreprocessor (KerasHub)
   ├─ ResNetBackbone
   └─ GlobalAveragePooling2D + Dense(2, softmax)
   │
   ▼
JSON: clase + confianza + probabilidades
```

## Modelo y contrato de datos

| Propiedad | Valor |
|---|---|
| Archivo | `models/ResNet50.keras` |
| Arquitectura | ResNet50 de KerasHub + cabeza de clasificación |
| Entrada | `224 × 224 × 3`, RGB, float32 |
| Salida | `Dense(2, activation="softmax")` |
| Índice 0 | `diseased` — Hoja enferma |
| Índice 1 | `healthy` — Hoja sana |
| TensorFlow | 2.21.0 |
| Keras | 3.13.2 |
| KerasHub | 0.26.0 |

El orden de clases corresponde a `train_ds.class_names == ['diseased', 'healthy']`. La entrada de entrenamiento se generó con `image_dataset_from_directory(..., image_size=(224, 224))`, cuyo ajuste predeterminado utiliza interpolación bilinear. La API convierte a RGB y redimensiona de la misma forma. No se aplica `rescale=1./255` ni otro `preprocess_input` en Python: el `ResNetImageClassifierPreprocessor` quedó serializado dentro del `.keras` y hace la transformación específica de ResNet.

## Endpoints

### `GET /`
Devuelve la interfaz para seleccionar una fotografía y consultar su clasificación.

### `GET /health`
Indica si la aplicación está activa y si se cargó el modelo. El campo `model_loaded` debe ser `true` antes de probar `/predict`.

### `POST /predict`
Recibe una imagen en el campo multipart `file` y devuelve la clase predicha, confianza y probabilidades. Ejemplo ilustrativo (no representa una inferencia medida):

```json
{
  "prediction": "diseased",
  "class": "Hoja enferma",
  "confidence": 0.94,
  "probabilities": {
    "diseased": 0.94,
    "healthy": 0.06
  }
}
```

Validaciones implementadas: archivo vacío (400), contenido que no sea imagen (415), imagen ilegible (400), archivo mayor al límite configurado (413) y modelo ausente (503).

## Ejecutar en Docker con el modelo incluido en el ZIP

No se requiere instalar Docker en la computadora personal si se usa GitHub Codespaces o un servicio PaaS que construya el Dockerfile.

Desde la raíz `limon-resnet50-api/` del proyecto:

```bash
docker build -t limon-resnet50-api .
docker run --rm -p 8000:8000 \
  -e MODEL_PATH=/app/models/ResNet50.keras \
  limon-resnet50-api
```

Abre `http://localhost:8000`, `http://localhost:8000/docs` y `http://localhost:8000/health`. También se puede ejecutar `docker compose up --build`; el archivo `compose.yaml` monta `models/` en modo de solo lectura.

## Guardar el modelo para un despliegue remoto

El archivo mide aproximadamente 90.4 MiB. Para no incorporar un binario grande al historial Git, `.gitignore` excluye `models/*.keras`. El ZIP de entrega sí contiene `models/ResNet50.keras`.

Para despliegue desde GitHub, crea un repositorio de modelos en Hugging Face Hub y carga allí `ResNet50.keras`. Después configura en el PaaS:

- `HF_REPO_ID`: `usuario/repositorio-modelo`.
- `HF_MODEL_FILENAME`: `ResNet50.keras`.
- `HF_TOKEN`: solo si el repositorio de modelos es privado.
- `IMAGE_SIZE`: `224`.
- `MAX_UPLOAD_MB`: `10`.

El servicio descarga el archivo al iniciar y lo carga una sola vez. No guardes `HF_TOKEN` en Git ni en el README. Si el repositorio es público, `HF_TOKEN` puede omitirse.

## GitHub Codespaces (trabajo desde el navegador)

1. Crea un repositorio en GitHub y sube el contenido del proyecto, excepto el `.keras` (se aloja en Hugging Face Hub para el despliegue remoto).
2. En GitHub selecciona **Code → Codespaces → Create codespace on main**.
3. Espera a que se prepare `.devcontainer/devcontainer.json` con Docker-in-Docker.
4. En la terminal ejecuta `docker version`.
5. Para probar la aplicación local dentro de Codespaces, carga el modelo desde Hugging Face mediante `HF_REPO_ID` o copia temporalmente el `.keras` a `models/`.
6. Ejecuta `docker build -t limon-resnet50-api .` y luego `docker run --rm -p 8000:8000 -e HF_REPO_ID="usuario/repositorio-modelo" limon-resnet50-api`.
7. Abre el puerto 8000 y valida `/`, `/health`, `/docs` y una predicción real.

Codespaces se ejecuta en infraestructura remota; revisa las cuotas/condiciones de uso de tu cuenta.

## Despliegue sugerido en Render

El Dockerfile escucha en `0.0.0.0` y respeta la variable `PORT` del PaaS.

1. Sube el código fuente a GitHub.
2. En Render elige **New → Web Service** y conecta el repositorio.
3. Elige **Docker** como entorno y deja `Dockerfile` como ruta.
4. En **Environment** configura `HF_REPO_ID`, `HF_MODEL_FILENAME=ResNet50.keras` y, si hace falta, `HF_TOKEN`.
5. Configura health check `/health` y crea el servicio.
6. Verifica `/health` (`model_loaded: true`), `/docs` y envía imágenes de prueba a `/predict`.

**Recursos:** Render Free tiene 512 MB de RAM y es para pruebas ligeras; no es una elección apropiada para asumir que TensorFlow + ResNet50 cargarán sin problemas. Como punto de partida, considera Render Standard (1 CPU, 2 GB RAM) y monitorea el consumo real; puede tener costo. La nomenclatura de planes cambió en agosto de 2026: Standard corresponde al plan `1c-2g`.

## Pruebas

Instala las dependencias del proyecto y ejecuta:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest httpx
pytest -q
```

La imagen de producción usa `tensorflow-cpu==2.21.0` para ejecución CPU. El paquete expone el módulo `tensorflow` y mantiene las versiones de Keras/KerasHub reportadas desde Colab.

Las pruebas HTTP incluyen un doble de prueba para verificar el contrato de respuesta. Una prueba adicional inspecciona el archivo `.keras` como ZIP y valida su estructura, pero **no sustituye la carga real del modelo ni una prueba de predicción**. Esas dos evidencias deben ejecutarse dentro de Codespaces/Docker con las dependencias instaladas.

## Estructura

```text
limon-resnet50-api/
├── app/
│   ├── main.py
│   ├── model_service.py
│   ├── settings.py
│   └── static/index.html
├── docs/
│   ├── MANUAL_DESPLIEGUE_NUBE.md
│   ├── INFORME_VALIDACION_PRUEBAS.md
│   └── REGISTRO_VERSIONES_MODELO.md
├── models/
│   ├── ResNet50.keras
│   └── MODEL_CARD.md
├── tests/test_api.py
├── .devcontainer/devcontainer.json
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── compose.yaml
├── README.md
└── requirements.txt
```

## Limitaciones

- No se reporta accuracy del servicio ni inferencia real en este documento; se deben medir con imágenes de prueba no utilizadas en entrenamiento/validación.
- La confianza softmax no es una garantía de diagnóstico agronómico.
- El prototipo no incluye autenticación, base de datos ni rate limiting.
- El contenedor puede ser pesado por TensorFlow. Monitoriza memoria y latencia tras desplegar.

## Referencias oficiales

- TensorFlow 2.21: compatibilidad de Python e instalación: https://www.tensorflow.org/install/pip
- TensorFlow CPU 2.21.0 y ruedas disponibles: https://pypi.org/project/tensorflow-cpu/2.21.0/
- KerasHub `ResNetImageClassifierPreprocessor`: https://keras.io/api/keras_hub/models/resnet/resnet_image_classifier_preprocessor/
- GitHub: límites de tamaño de archivos: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- Render: planes de cómputo: https://render.com/docs/compute-plans
- Render: limitaciones del servicio Free: https://render.com/docs/free
