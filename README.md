# Proyecto_final_740

# Transferencia de Aprendizaje con Fine-Tuning en TensorFlow / Python

## Introducción

La **Transferencia de Aprendizaje** es una técnica en el aprendizaje automático que permite aprovechar modelos pre-entrenados en grandes conjuntos de datos, como ImageNet, para tareas específicas con conjuntos de datos más pequeños. Esto es útil especialmente cuando se dispone de recursos limitados y se desea aprovechar modelos avanzados sin entrenarlos desde cero.

El **Fine-Tuning** (afinamiento) es el proceso de ajustar un modelo pre-entrenado para realizar una tarea más específica, desbloqueando algunas de sus capas para que sus pesos puedan ser actualizados durante el entrenamiento en el nuevo conjunto de datos.

## Conceptos Clave

1. **Transferencia de Aprendizaje (Transfer Learning):**
   - En el aprendizaje profundo, los modelos pre-entrenados han aprendido representaciones generalizadas útiles de los datos. Por ejemplo, un modelo entrenado en imágenes generales, como ImageNet, aprende a identificar características como bordes, texturas y formas que son relevantes para múltiples tareas de clasificación de imágenes.
   - En lugar de entrenar un modelo desde cero, la transferencia de aprendizaje utiliza estos modelos pre-entrenados como un punto de partida, y se ajustan (fine-tuning) a una tarea más específica.

2. **Fine-Tuning:**
   - El fine-tuning ajusta un modelo pre-entrenado para que pueda realizar una tarea específica. Esto se hace desbloqueando algunas de las capas más profundas del modelo y permitiendo que sus pesos se actualicen mientras se entrena en un nuevo conjunto de datos.
   - Un ejemplo común es utilizar un modelo pre-entrenado en la clasificación general de imágenes (como perros y gatos) y luego ajustarlo para clasificar algo más específico, como diferentes razas de perros.

## Ventajas del Fine-Tuning

- **Menos Datos y Tiempo de Entrenamiento:** Fine-tuning suele requerir menos datos y tiempo de cómputo que entrenar un modelo desde cero.
- **Aprovechamiento de Modelos Avanzados:** Permite usar modelos de última generación (como MobileNet, ResNet, GPT-3) para tareas específicas, sin la necesidad de grandes cantidades de datos o recursos computacionales.
- **Mejores Resultados:** Los modelos pre-entrenados han aprendido representaciones útiles que son generalmente aplicables a una amplia gama de tareas, lo que puede mejorar el rendimiento en tareas con datos limitados.

## Implementación de Fine-Tuning con TensorFlow

A continuación, se presenta un ejemplo de cómo realizar **Fine-Tuning** usando un modelo pre-entrenado MobileNetV2 de TensorFlow para clasificar imágenes de flores.

### Pasos del Código

1. **Importación de Bibliotecas y Carga del Modelo Pre-entrenado:**
   - Se importan las bibliotecas necesarias como TensorFlow, TensorFlow Hub, y `ImageDataGenerator` para preprocesar las imágenes.
   - Se utiliza un modelo pre-entrenado de MobileNetV2 desde TensorFlow Hub para realizar la clasificación de imágenes.

2. **Cargar Datos de Imágenes:**
   - Se descarga el conjunto de datos de imágenes de flores, que incluye diferentes tipos de flores. Se usa `ImageDataGenerator` para cargar las imágenes y escalarlas para el modelo.

3. **Predicción con el Modelo Pre-entrenado:**
   - El modelo realiza predicciones sobre un conjunto de imágenes de flores sin realizar ningún fine-tuning. Se muestra cómo el modelo puede clasificar las imágenes con las etiquetas correspondientes.

4. **Mostrar Resultados:**
   - Se muestran las imágenes junto con sus predicciones en una figura utilizando `matplotlib`.

### Código Ejemplo

```python
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# URL del modelo MobileNetV2 pre-entrenado
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"

# Directorio de datos de flores
data_root = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

# Crear un generador de datos de imágenes para el conjunto de datos de flores
image_data_generator = ImageDataGenerator(rescale=1/255)
image_data = image_data_generator.flow_from_directory(data_root, target_size=(224, 224), batch_size=1)

# Cargar el modelo MobileNetV2 desde TensorFlow Hub
model = hub.load(model_url)

# Contador para limitar a 10 imágenes
count = 0
num_images = 10  # Número de imágenes para mostrar

# Listas para almacenar las imágenes y las clasificaciones
images = []
classifications = []

# Procesar imágenes de flores y obtener las predicciones
for batch in image_data:
    image = batch[0]  # Obtener la imagen del lote
    predictions = model(image)  # Realizar predicciones
     
    # Obtener las etiquetas y probabilidades
    labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    labels_path = tf.keras.utils.get_file("ImageNetLabels.txt", labels_url)
    with open(labels_path) as file:
        labels = file.read().splitlines()
     
    top_label_index = tf.argmax(predictions, axis=-1)
    top_label = labels[top_label_index[0]]
     
    images.append(image[0])
    classifications.append(top_label)
     
    count += 1
    if count >= num_images:
        break  # Detener después de procesar el número deseado de imágenes

# Crear una matriz de imágenes y clasificaciones
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
axes = axes.ravel()

for i in range(num_images):
    axes[i].imshow(images[i])
    axes[i].set_title(classifications[i])
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

### Explicación del Código

1. **Cargar el Modelo:**
   - El modelo pre-entrenado `MobileNetV2` es cargado usando la URL proporcionada de TensorFlow Hub. Este modelo fue entrenado previamente con imágenes de ImageNet.

2. **Cargar y Preprocesar Imágenes:**
   - Usamos `ImageDataGenerator` para leer imágenes desde un directorio y normalizarlas (escalar los valores de píxeles a un rango de 0 a 1).
   - El conjunto de imágenes se redimensiona a 224x224 píxeles, que es el tamaño de entrada esperado por MobileNetV2.

3. **Realizar Predicciones:**
   - El modelo realiza predicciones sobre las imágenes cargadas. Las predicciones son probabilidades que indican qué clase (de las 1000 clases de ImageNet) es la más probable para cada imagen.

4. **Visualizar Resultados:**
   - Las imágenes junto con las predicciones se muestran usando `matplotlib` para verificar cómo el modelo clasifica las flores en el conjunto de datos.

## Conclusiones

- Este ejemplo muestra cómo usar un modelo pre-entrenado en un nuevo conjunto de datos sin realizar fine-tuning. El modelo puede hacer predicciones incluso sin ser ajustado específicamente para el conjunto de datos de flores, gracias a las características aprendidas en el entrenamiento con un conjunto de datos más grande.
- La transferencia de aprendizaje con fine-tuning es un enfoque poderoso cuando se tiene un conjunto de datos limitado, ya que permite usar modelos avanzados sin necesidad de entrenarlos desde cero, lo cual es más costoso en términos de tiempo y recursos computacionales.
