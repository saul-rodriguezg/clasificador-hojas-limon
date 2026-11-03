# Registro de versiones y trazabilidad del modelo

| Elemento | Valor registrado |
|---|---|
| Artefacto | `models/ResNet50.keras` |
| Tamaño | 94,781,285 bytes (90.4 MiB aprox.) |
| SHA-256 | `c277785b48f3b05ff877043559b466c0498e1bd2e279418cd24376633a409995` |
| Fecha de guardado (metadata Keras) | 2026-11-02 23:10:51 (según `metadata.json`) |
| TensorFlow reportado desde Colab | 2.21.0 |
| Keras reportado desde Colab | 3.13.2 |
| KerasHub reportado desde Colab | 0.26.0 |
| Runtime de contenedor | Python 3.11 + TensorFlow CPU 2.21.0 |
| Entrada | 224×224×3 |
| Salida | 2 logits softmax |
| Clases | 0=`diseased`, 1=`healthy` |

## Notas de integración

El archivo contiene las capas serializadas `ResNetImageClassifierPreprocessor`, `ResNetImageConverter`, `ResNetBackbone`, `GlobalAveragePooling2D` y `Dense(2, softmax)`. La API importa KerasHub antes de ejecutar `keras.saving.load_model(..., compile=False)` para registrar las clases serializadas. No se aplica una normalización adicional fuera del modelo.

La dependencia `tensorflow-cpu==2.21.0` se utiliza en el contenedor porque el despliegue objetivo es CPU. La distribución proporciona el módulo importable `tensorflow`; Keras y KerasHub mantienen las versiones reportadas en el notebook. La compatibilidad de carga/inferencia debe confirmarse ejecutando `pytest` e iniciando la API dentro del contenedor.
