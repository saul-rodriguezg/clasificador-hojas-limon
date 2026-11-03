# Manual técnico de despliegue en la nube

## 1. Objetivo y alcance

Desplegar una aplicación contenerizada que permite cargar una fotografía de una hoja de limón desde un navegador. La interfaz envía la imagen a una API REST implementada con FastAPI, que utiliza el modelo ResNet50 serializado en `ResNet50.keras` para devolver una clasificación `diseased` o `healthy`.

El proyecto se entrega con el artefacto del modelo en el ZIP. Para separar el código del binario durante el despliegue desde GitHub, se recomienda alojar el `.keras` en Hugging Face Hub y descargarlo al iniciar el contenedor.

## 2. Componentes y arquitectura

| Componente | Responsabilidad | Tecnología |
|---|---|---|
| Frontend | Seleccionar imagen y mostrar predicción | HTML, CSS y JavaScript |
| Backend | Validar archivos, exponer endpoints y coordinar inferencia | FastAPI / Uvicorn |
| Modelo | Clasificar la imagen en dos clases | ResNet50 + KerasHub |
| Serialización | Empaquetar arquitectura, pesos y preprocesamiento | Keras `.keras` |
| Contenedor | Empaquetar aplicación, dependencias y runtime | Docker |
| Repositorio | Versionar código y documentación | GitHub |
| Artefacto remoto | Distribuir el modelo para el contenedor cloud | Hugging Face Hub |
| PaaS | Construir Dockerfile y publicar el servicio HTTPS | Render Web Service |

Flujo de solicitud: navegador → `POST /predict` → FastAPI → preparación RGB 224×224 → ResNet50 con su preprocesador integrado → JSON de clase y probabilidades.

## 3. Requerimientos técnicos

- Python 3.11.
- TensorFlow CPU 2.21.0, Keras 3.13.2 y KerasHub 0.26.0.
- FastAPI, Uvicorn, Pillow, NumPy y python-multipart (versiones delimitadas en `requirements.txt`).
- Docker para construir/ejecutar; puede utilizarse GitHub Codespaces para no instalar Docker localmente.
- Repositorio GitHub.
- Cuenta Hugging Face para alojar `ResNet50.keras` si el código se despliega desde GitHub sin incluir el binario.
- Cuenta Render para crear el servicio web Docker.
- Como punto de partida de recursos para inferencia CPU: 1 CPU y 2 GB RAM. Medir el uso real en el servicio desplegado y aumentar memoria si hay reinicios por OOM.

El modelo pesa aproximadamente 90.4 MiB. La imagen de TensorFlow también es considerable; la construcción puede tardar y el almacenamiento de imágenes Docker debe contemplarse.

## 4. Preparación del modelo

El artefacto incluido se llama `models/ResNet50.keras`. La inspección estructural confirmó:

- entrada `(None, 224, 224, 3)`;
- capa de salida `Dense(2, activation="softmax")`;
- `ResNetImageClassifierPreprocessor` y `ResNetImageConverter` de KerasHub serializados dentro del grafo;
- `ResNetBackbone` tipo ResNet con bloques `[3, 4, 6, 3]`.

El orden de clases recuperado del conjunto de entrenamiento es `['diseased', 'healthy']`, por lo que índice 0 corresponde a hoja enferma e índice 1 a hoja sana.

La API replica la entrada del entrenamiento: convierte a RGB y cambia el tamaño a 224×224 con interpolación bilinear, como `image_dataset_from_directory` con `image_size=(224,224)`. No aplica normalización adicional, porque el preprocesador de KerasHub ya está incluido en el modelo.

### Publicar el modelo en Hugging Face Hub

1. Inicia sesión en Hugging Face.
2. Crea un repositorio de tipo **Model**. Se puede mantener privado.
3. Sube `ResNet50.keras` conservando exactamente ese nombre.
4. Copia el identificador `usuario/repositorio-modelo`.
5. Si el repositorio es privado, crea un token con permisos de lectura para descargar el modelo.

No incluyas el token dentro del código ni en GitHub.

## 5. Preparar GitHub

1. Crea un repositorio nuevo, por ejemplo `clasificador-hojas-limon-resnet50`.
2. Sube el contenido de la carpeta `limon-resnet50-api/` a la raíz del repositorio.
3. Conserva `Dockerfile`, `requirements.txt`, `app/`, `tests/`, `docs/`, `.devcontainer/` y `README.md`.
4. El `.gitignore` excluye `models/*.keras`; esto evita registrar el binario de 90.4 MiB en el historial Git. El ZIP de entrega sí incluye el modelo.
5. No subas `.env` ni tokens. El archivo `.env.example` solo muestra nombres de variables.

## 6. Construir y validar Docker desde el navegador

GitHub Codespaces permite trabajar dentro de un entorno remoto de desarrollo y el proyecto incluye configuración Docker-in-Docker.

1. Abre el repositorio GitHub.
2. Selecciona **Code → Codespaces → Create codespace on main**.
3. Cuando se abra el entorno, ejecuta `docker version` en la terminal.
4. Para probar la imagen con el modelo remoto, configura temporalmente las variables:

```bash
export HF_REPO_ID="usuario/repositorio-modelo"
export HF_MODEL_FILENAME="ResNet50.keras"
# Para repositorio privado:
export HF_TOKEN="hf_..."
```

5. Construye la imagen:

```bash
docker build -t limon-resnet50-api .
```

6. Ejecuta el contenedor:

```bash
docker run --rm -p 8000:8000 \
  -e HF_REPO_ID="$HF_REPO_ID" \
  -e HF_MODEL_FILENAME="$HF_MODEL_FILENAME" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MODEL_PATH=/app/models/ResNet50.keras \
  limon-resnet50-api
```

7. Abre el puerto reenviado 8000 y comprueba `/`, `/health` y `/docs`.
8. Confirma que `/health` devuelva `model_loaded: true`; después envía una fotografía a `/predict`.

Si haces la prueba con el archivo local incluido en el ZIP, abre el proyecto que contiene `models/ResNet50.keras` y ejecuta `docker build`; el Dockerfile copia el directorio `models/` a la imagen.

## 7. Despliegue en Render

1. En Render, selecciona **New → Web Service**.
2. Conecta el repositorio GitHub.
3. Selecciona **Docker** como entorno de ejecución y confirma `Dockerfile` como ruta.
4. Configura estas variables en el panel de Render:

| Variable | Valor |
|---|---|
| `HF_REPO_ID` | `usuario/repositorio-modelo` |
| `HF_MODEL_FILENAME` | `ResNet50.keras` |
| `HF_TOKEN` | Token de lectura, solo si el modelo es privado |
| `IMAGE_SIZE` | `224` |
| `MAX_UPLOAD_MB` | `10` |

5. Configura el health check en `/health`.
6. Para este modelo, no selecciones un plan de 512 MB como destino final. Render identifica el plan Standard como 1 CPU y 2 GB RAM (`1c-2g`); es un punto de partida para la prueba, no una garantía de capacidad. Monitoriza memoria, tiempos de arranque y latencia; aumenta recursos si es necesario.
7. Inicia el despliegue y revisa logs de build y arranque.

El servicio escucha en `0.0.0.0` y utiliza el puerto provisto por `PORT`. Render puede reconstruir el contenedor al detectar nuevos commits en la rama configurada.

## 8. Validación posterior al despliegue

Con la URL HTTPS que proporciona Render:

1. `GET /`: la interfaz responde y permite seleccionar una imagen.
2. `GET /health`: estado `ok`, `model_loaded: true` y clases `['diseased','healthy']`.
3. `GET /docs`: Swagger muestra `GET /health` y `POST /predict`.
4. `POST /predict`: subir una imagen JPG o PNG de hoja y confirmar que la respuesta contiene `prediction`, `class`, `confidence` y las probabilidades de ambas clases.
5. Repetir con ejemplos sanos y enfermos separados de los datos de entrenamiento y validación.
6. Guardar capturas de pantalla, fecha, commit desplegado y resultados en el informe de pruebas.

## 9. Diagnóstico

| Síntoma | Posibles causas y acción |
|---|---|
| `model_loaded: false` | Revisar `HF_REPO_ID`, nombre exacto `ResNet50.keras`, token y logs de descarga. |
| Error al deserializar `.keras` | Verificar que se instalen las versiones TensorFlow/Keras/KerasHub registradas. |
| `POST /predict` devuelve 503 | El modelo no terminó de cargar; revisar `/health` y logs. |
| Reinicio por memoria | Aumentar RAM del servicio y revisar métricas de memoria. |
| Error 400 al subir foto | Verificar que el archivo sea una imagen válida y no esté truncado. |
| Error 413 | Reducir la imagen o subir `MAX_UPLOAD_MB` conscientemente. |

## 10. Seguridad y buenas prácticas

- Mantener secretos como variables de entorno del PaaS.
- Mantener el contenedor ejecutándose con usuario no root.
- Limitar tamaño de subida.
- No exponer tokens en capturas ni documentación.
- Registrar el hash SHA-256 del artefacto para comprobar integridad.
- Mantener `requirements.txt` y la imagen base versionados.
- No interpretar la confianza softmax como certeza clínica/agronómica.

## 11. Evidencias que deben añadirse tras el despliegue real

| Evidencia | Estado al preparar este manual |
|---|---|
| Código fuente y Dockerfile | Incluidos en ZIP |
| Archivo `.keras` | Incluido en ZIP |
| Construcción Docker con logs | Pendiente de ejecutar en Codespaces o Render |
| Contenedor en ejecución | Pendiente |
| `/health` con `model_loaded: true` | Pendiente |
| Predicciones reales | Pendiente |
| URL pública Render | Pendiente |
| Capturas de Swagger y frontend | Pendiente |

