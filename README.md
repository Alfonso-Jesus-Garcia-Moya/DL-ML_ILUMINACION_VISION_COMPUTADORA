Tema 4. ILUMINACIÓN
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
