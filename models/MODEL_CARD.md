# Artefacto del modelo

- Archivo incluido en el paquete de entrega: `ResNet50.keras`
- Arquitectura: ResNet50 de KerasHub con clasificador `Dense(2, softmax)`
- Entrada: `(None, 224, 224, 3)`
- Salida: `(None, 2)`
- Orden de clases: índice 0 = `diseased`; índice 1 = `healthy`
- Preprocesamiento: capa `ResNetImageClassifierPreprocessor` serializada dentro del modelo
- TensorFlow: 2.21.0
- Keras: 3.13.2
- KerasHub: 0.26.0

El archivo `.keras` se excluye del control de versiones Git para no añadir un binario grande al historial del repositorio. Para el despliegue remoto se debe cargar a un repositorio de modelos en Hugging Face Hub y configurar `HF_REPO_ID` y `HF_MODEL_FILENAME`. El ZIP de entrega sí contiene el archivo del modelo.
