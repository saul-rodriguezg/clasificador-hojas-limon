# Documento de validación y pruebas

## 1. Objetivo

Verificar la integración entre frontend, API REST, archivo del modelo ResNet50 y contenedor Docker para la clasificación de hojas de limón en `diseased` y `healthy`.

## 2. Identificación del artefacto

| Elemento | Valor |
|---|---|
| Archivo del modelo | `models/ResNet50.keras` |
| Tamaño | 94,781,285 bytes (aprox. 90.4 MiB) |
| SHA-256 | `c277785b48f3b05ff877043559b466c0498e1bd2e279418cd24376633a409995` |
| Arquitectura | ResNet50 de KerasHub + Dense(2, softmax) |
| Entrada | `(None, 224, 224, 3)` |
| Salida | `(None, 2)` |
| Clases | `diseased` (0), `healthy` (1) |
| TensorFlow | 2.21.0 (distribución CPU en el contenedor) |
| Keras | 3.13.2 |
| KerasHub | 0.26.0 |
| API | FastAPI |
| Contenedor | Docker |

La inspección del archivo `.keras` confirmó que incluye un `ResNetImageClassifierPreprocessor` y `ResNetImageConverter` de KerasHub, un `ResNetBackbone` con bloques ResNet [3,4,6,3], pooling global y una capa Dense de dos salidas softmax.

## 3. Estrategia de pruebas

Se separan tres niveles de validación para no afirmar resultados que todavía no se han medido:

1. **Pruebas estructurales:** inspección del ZIP `.keras` y validación de dimensiones, capa de salida y preprocesador serializado.
2. **Pruebas de API con doble de prueba:** ejercitan el contrato HTTP, respuesta JSON, conversiones de imagen y rechazos de archivos sin depender del tiempo de inferencia.
3. **Pruebas de integración pendientes:** carga real del `.keras`, inferencia sobre fotografías de evaluación, construcción Docker y despliegue público.

## 4. Casos de prueba automatizados incluidos

| ID | Tipo | Caso | Resultado esperado | Estado |
|---|---|---|---|---|
| P-01 | Funcional | `GET /` | HTTP 200 e interfaz HTML | Incluida en pytest |
| P-02 | Funcional | `GET /health` | HTTP 200 e información de modelo/clases | Incluida en pytest |
| P-03 | API | `POST /predict` con doble de prueba | HTTP 200, clase y probabilidades | Incluida en pytest |
| P-04 | Validación | Subir un archivo no imagen | HTTP 415 | Incluida en pytest |
| P-05 | Extremo | Subir bytes corruptos como imagen | HTTP 400 | Incluida en pytest |
| P-06 | Extremo | Subir archivo vacío | HTTP 400 | Incluida en pytest |
| P-07 | Extremo | Subir archivo mayor al límite | HTTP 413 | Incluida en pytest |
| P-08 | Disponibilidad | Solicitar predicción sin modelo cargado | HTTP 503 | Incluida en pytest |
| P-09 | Imagen | Imagen en escala de grises | Se convierte a RGB 224×224 | Incluida en pytest |
| P-10 | Imagen | Imagen RGBA/transparente | Se convierte a RGB 224×224 | Incluida en pytest |
| P-11 | Artefacto | Inspeccionar `ResNet50.keras` | Dense(2, softmax), preprocesador 224×224 | Incluida en pytest |
| P-12 | Integración | Cargar modelo con TensorFlow/Keras/KerasHub fijados | `model_loaded: true` | Pendiente de ejecutar |
| P-13 | Integración | Inferencia de fotografía real | JSON válido con dos probabilidades | Pendiente de ejecutar |
| P-14 | Docker | Construir imagen | `docker build` termina con código 0 | Pendiente de ejecutar |
| P-15 | Docker | Ejecutar contenedor | `/health` responde desde el contenedor | Pendiente de ejecutar |
| P-16 | Cloud | Abrir URL pública | HTTPS disponible | Pendiente de ejecutar |
| P-17 | Cloud | Predicción con modelo remoto | Inferencia real en PaaS | Pendiente de ejecutar |

## 5. Verificación estructural ya realizada sobre el archivo

La inspección de `ResNet50.keras` (archivo ZIP con `config.json`, `metadata.json` y `model.weights.h5`) produjo estos resultados:

- `metadata.json` reporta Keras 3.13.2.
- La entrada del modelo tiene forma `[None, 224, 224, 3]`.
- La salida corresponde a la capa `Dense` con 2 unidades y activación `softmax`.
- La capa `ResNetImageClassifierPreprocessor` está serializada en el modelo.
- El convertidor almacena tamaño `[224,224]`, escala y offset por canal, interpolación `bicubic`, y opciones de ajuste de aspecto.
- El orden de clases proviene del notebook: `['diseased', 'healthy']`.
- SHA-256 calculado: `c277785b48f3b05ff877043559b466c0498e1bd2e279418cd24376633a409995`.

Esta inspección acredita la estructura del artefacto, pero no acredita que una inferencia real se haya ejecutado correctamente.

## 6. Resultados de ejecución del entorno de preparación

El entorno de preparación disponible tiene FastAPI, Pillow, NumPy y pytest, pero no tiene instalados TensorFlow, Keras ni KerasHub y no dispone de Docker. La instalación de dependencias de TensorFlow desde este entorno no pudo completarse por falta de conexión al índice de paquetes.

Por eso:

- Las pruebas de API con clasificador simulado verifican el contrato HTTP, no la precisión ni la carga del modelo real.
- La prueba estructural valida el archivo `.keras` sin ejecutarlo.
- No se declara como aprobada una construcción de imagen Docker ni una predicción real.

Después de ejecutar en Codespaces/Docker, registrar aquí la salida real de `pytest -q`, `docker build`, `docker run`, `/health` y una predicción. Agregar fecha, URL pública y capturas.

## 7. Procedimiento para cerrar la validación

1. En Codespaces, instalar dependencias: `pip install -r requirements.txt pytest httpx`.
2. Ejecutar `pytest -q` y guardar la salida completa.
3. Construir con `docker build -t limon-resnet50-api .`.
4. Ejecutar el contenedor con el archivo local o configurar `HF_REPO_ID` y `HF_MODEL_FILENAME`.
5. Consultar `/health`; continuar solo si `model_loaded` es `true`.
6. Subir fotografías de hojas sanas y enfermas no usadas para entrenar.
7. Registrar predicción, clase real, confianza, latencia y matriz de confusión si se dispone de suficientes ejemplos etiquetados.
8. Desplegar en Render y repetir las pruebas mediante la URL HTTPS.

## 8. Conclusión

El artefacto ResNet50 se recibió y su estructura coincide con la interfaz esperada por la aplicación: imágenes 224×224 RGB y dos salidas softmax en el orden `diseased`, `healthy`. La API tiene casos automatizados para el contrato HTTP y entradas límite. La validación final aún requiere ejecutar el modelo con sus dependencias exactas, construir el contenedor y comprobar el despliegue. No se atribuye al modelo una precisión nueva ni se presentan predicciones de ejemplo como resultados reales.
