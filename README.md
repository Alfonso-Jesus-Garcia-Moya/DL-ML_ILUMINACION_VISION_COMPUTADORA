Tema 4. ILUMINACIÓN parte 1 de 3 , abajo parte 2
4.1. Importancia de la iluminación en visión por computadora.

4.2. Problemas relacionados con la iluminación.

4.3. Preprocesamiento de imágenes.

4.4. Aumento de datos específico para la iluminación.

READMEN DE REPORTE
# 🐕 Proyección de Imágenes con PCA y UMAP (Stanford Dogs Dataset)

## 📌 Tarea de la Semana 9: Análisis Visual de Dimensionalidad

Este proyecto aplica técnicas de **reducción de dimensionalidad** (PCA y UMAP) sobre un subconjunto del **Stanford Dogs Dataset** para visualizar cómo se agrupan las diferentes razas en un espacio de baja dimensión (2D y 3D), después de un preprocesamiento de imágenes que incluye aumento de iluminación.

### 🎯 Objetivo

Visualizar la estructura latente de las representaciones de imágenes mediante técnicas lineales y no lineales, demostrando la robustez de las características de la imagen frente a variaciones de iluminación.

### 🛠️ Pipeline de Procesamiento

1.  **Carga del Dataset:** Extracción del conjunto de imágenes del Stanford Dogs Dataset.
2.  **Aumento de Iluminación:** Aplicación de variaciones aleatorias de **Brillo ($\beta$)** y **Contraste ($\alpha$)** a cada imagen para simular diversas condiciones de luz y mejorar la robustez.
    $$\text{Imagen Ajustada} = \alpha \cdot \text{Imagen Original} + \beta$$
3.  **Conversión y Aplanamiento:**
    * Redimensión a $128 \times 128 \times 3$ y normalización a $[0, 1]$.
    * Aplanamiento del tensor 4D a una matriz de vectores de características de alta dimensión.
4.  **Reducción de Dimensionalidad:** Proyección de los vectores a 3 dimensiones utilizando:
    * **PCA (Análisis de Componentes Principales):** Método lineal que maximiza la varianza.
    * **UMAP (Uniform Manifold Approximation and Projection):** Método no lineal que preserva la estructura topológica local.

### 📊 Resultados y Análisis

Los resultados se visualizan mediante gráficos de dispersión 2D y 3D, donde cada punto representa una imagen y el color indica la raza.

* **PCA:** Muestra una **superposición significativa** de las razas, lo que sugiere que las características distintivas de las razas no son linealmente separables en los primeros componentes principales.
* **UMAP:** Logra una **mejor segregación y clústeres más compactos**, demostrando su capacidad para capturar las relaciones no lineales y la estructura intrínseca del *manifold* de las imágenes.

### 📦 Tecnologías Utilizadas

* `Python 3.x`
* `scikit-learn` (para PCA)
* `umap-learn` (para UMAP)
* `OpenCV (cv2)` (para Preprocesamiento de imágenes)
* `matplotlib`, `seaborn`, `plotly` (para Visualización)
* `numpy`

### 🚀 Uso

1.  Clonar el repositorio.
2.  Asegurar el archivo `perros.zip` del Stanford Dogs Dataset en la ruta de trabajo.
3.  Ejecutar el *notebook* de Colab o Jupyter.


PARTE 2 DE 3
# 💡 DEEP LEARNIN & ML_ILUMINACION_VISION_COMPUTADORA

## 📄 Semana 10: Fundamentos de CNN, Convolución y Pooling

Este repositorio contiene el código desarrollado y ejecutado para la demostración de los componentes fundamentales de las Redes Neuronales Convolucionales (CNN) y su aplicación inicial en el contexto de la visión por computadora.

El enfoque principal de este notebook es la **validación conceptual** de cómo los modelos procesan datos espaciales (imágenes), desde la unidad más básica (el perceptrón) hasta las operaciones clave de una capa convolucional.

### 🛠️ Contenido del Notebook

El archivo `Semana_10_Con tarea.ipynb` incluye las siguientes demostraciones prácticas:

1.  **Perceptrón y Separabilidad Lineal:** Implementación y entrenamiento de un perceptrón simple para resolver la compuerta lógica AND, incluyendo la visualización de la frontera de decisión.
2.  **Convolución 2D Fundamental:** Demostración manual de la operación de convolución utilizando una imagen artificial y un kernel detector de bordes, ilustrando el proceso de generación de mapas de características.
3.  **Pipeline de Procesamiento CNN en Imágenes Reales:** Aplicación de filtros de convolución y la operación de Max Pooling sobre imágenes del Stanford Dogs Dataset (o imágenes de ejemplo) para simular la extracción de características y la reducción de dimensionalidad espacial:
    * Detección de Bordes (Filtro Sobel/Laplaciano).
    * Suavizado (*Blur*).
    * Max Pooling Iterativo (generación de una jerarquía de características abstractas).

### 🎯 Objetivos de Aprendizaje

* Comprender el rol de la neurona y las funciones de activación (ReLU, Sigmoide).
* Visualizar la **invariancia traslacional** lograda mediante el compartimiento de pesos en la convolución.
* Analizar cómo la operación de **Max Pooling** reduce la resolución mientras conserva las características dominantes, contribuyendo a la robustez del modelo.

### 📦 Dependencias

* numpy
* matplotlib
* opencv-python (cv2)


PARTE 3 DE 3

# 🐶 Clasificación Avanzada de Imágenes con Deep Learning (Stanford Dogs Dataset)



[Image of a deep learning model architecture for image classification]


Este repositorio aborda la desafiante tarea de la **Clasificación de Grano Fino (Fine-Grained Classification)** utilizando el **Stanford Dogs Dataset** (120 razas de perros). El objetivo principal es la implementación, el entrenamiento y el **análisis comparativo** de distintas arquitecturas de Deep Learning (DL) y la técnica de Transferencia de Conocimiento.

---

## 💡 Objetivos y Enfoque

El proyecto busca evaluar y contrastar el **rendimiento** y la **eficiencia** de diversos modelos neuronales en un contexto de alta complejidad visual, donde las diferencias entre las clases (razas) son sutiles.

### Arquitecturas a Evaluar

1.  **MLP (Línea Base):** Para establecer el rendimiento sin estructura espacial.
2.  **LSTM Bidireccional:** Evalúa la imagen como una secuencia de datos.
3.  **CNN Baseline:** El estándar para Computer Vision, extrayendo jerarquías de características locales.
4.  **Transfer Learning (MobileNetV2):** Reutilización de conocimiento preentrenado en ImageNet para máxima precisión y rápida convergencia.

---

## 🛠️ Estructura del Pipeline de Entrenamiento

El código sigue un protocolo de experimentación riguroso, optimizado para la **reproducibilidad** y el rendimiento, utilizando TensorFlow y Keras.

### 1. Preparación y Optimización del Dataset

* **Reproducibilidad:** Se utiliza una semilla fija (`SEED = 42`) para garantizar la consistencia en la división de datos y la inicialización de pesos.
* **Pipeline de Datos (`tf.data`):** Implementación de técnicas avanzadas como **caching**, **prefetching** y **shuffling** para maximizar el rendimiento de la GPU/CPU durante el entrenamiento.
* **Normalización:** Escalado de las imágenes al rango $[0, 1]$.
* **Adaptación de Tensors:** Funciones específicas para reestructurar la entrada según la arquitectura:
    * **MLP:** Imagen aplanada a vector 1D.
    * **LSTM:** Imagen transformada a una secuencia de filas (`Tiempo x Características`).
    * **CNN/TL:** Mantenimiento de la forma espacial (`H x W x C`).

### 2. Control del Aprendizaje (Callbacks)

Se emplean Callbacks para gestionar el proceso de entrenamiento de forma automática y robusta:

* `EarlyStopping`: Detiene el entrenamiento al detectar **sobreajuste** (monitoreando la pérdida de validación).
* `ReduceLROnPlateau`: Ajusta dinámicamente la **tasa de aprendizaje** para mejorar la convergencia en etapas finales.
* `ModelCheckpoint`: Guarda la versión del modelo que alcanza el **mejor desempeño** en el conjunto de validación.

### 3. Comparativa de Arquitecturas

El corazón del proyecto es la comparación de los cuatro enfoques:

| Modelo | Tipo de Estructura | Función en el Análisis |
| :--- | :--- | :--- |
| **MLP (Línea Base)** | Vectorial (Densas) | Evalúa el desempeño sin estructura espacial. |
| **LSTM Bidireccional** | Secuencial/Temporal | Evalúa la imagen como una secuencia de filas. |
| **CNN Baseline** | Espacial (Convolucional) | Estándar para CV; extrae jerarquías de características locales. |
| **Transfer Learning** | Preentrenado + Fine-Tuning | Máxima precisión y rápida convergencia. |

---

## 📊 Evaluación Profunda

El código incluye un conjunto de funciones de evaluación avanzadas para una visión integral del rendimiento del modelo:

* `plot_history`: Visualización de curvas de **Pérdida (Loss)** y **Precisión (Accuracy)** en entrenamiento y validación.
* `eval_and_report`: Generación del **Reporte de Clasificación** (Precisión, Recall, F1-Score por clase) y la **Matriz de Confusión** (`heatmap`).
* `show_sample_predictions`: Muestras visuales de las predicciones (**aciertos ✅ / errores ❌**) para un análisis cualitativo.



[Image of a confusion matrix heatmap]


---

## 🚀 Tecnologías Clave

* **Python 3.x**
* **TensorFlow / Keras** (Core DL framework)
* **tf.data** (Optimización de pipelines de datos)
* **numpy**
* **matplotlib** y **seaborn** (Visualización de resultados)
* **scikit-learn** (Generación de Reportes y Matriz de Confusión)
