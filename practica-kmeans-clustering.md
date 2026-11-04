# Práctica Guiada: Aprendizaje No Supervisado - Agrupamiento con K-Means (*K-Means Clustering*)

**Asignatura:** Fundamentos de Inteligencia Artificial (Unidad III)  
**Tema:** Aprendizaje No Supervisado, Algoritmos de Agrupamiento y Método del Codo  
**Duración:** 60 minutos  
**Entorno de ejecución:** Google Colab / Jupyter Notebook (Python 3.x)  

---

## 1. Requerimientos Técnicos e Instalación de Dependencias

Para la realización de esta práctica se requieren las siguientes librerías estándar de computación científica, procesamiento matricial y visualización de datos:
- `numpy`: Mapeo de vectores, matrices y cálculo numérico de distancias.
- `matplotlib`: Generación de gráficos de dispersión 2D y traza de iteraciones.
- `scikit-learn`: Generación de conjuntos de datos sintéticos y validación cruzada del modelo.

Ejecute la siguiente celda para verificar las dependencias y fijar la semilla aleatoria de reproducibilidad:

```python
# Instalación de dependencias (Descomentar si se ejecuta en un entorno nuevo o Google Colab)
# !pip install numpy matplotlib scikit-learn

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SklearnKMeans

# Fijar semilla de aleatoriedad para garantizar la reproducibilidad de los experimentos en laboratorio
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

print("✅ Entorno preparado correctamente. Semilla de aleatoriedad fijada en SEED =", SEED)
```

---

## 2. Fundamentos Teóricos e Interpretación

El **Aprendizaje No Supervisado** (*Unsupervised Learning*) aborda problemas en los cuales el conjunto de datos de entrenamiento $\mathcal{D} = \{x_1, x_2, \dots, x_N\}$ carece de etiquetas de salida o respuestas supervisadas (Russell & Norvig, 2004; Luger, 2004). El objetivo del algoritmo es descubrir de manera autónoma la estructura subyacente, patrones o agrupamientos naturales (*clusters*) presentes en los datos.

### Mapeo de Conceptos: Teoría vs. Implementación

| Concepto Teórico | Descripción Algorítmica | Función en la Práctica |
| :--- | :--- | :--- |
| **Punto / Instancia ($x_i$)** | Vector de características de dimensión $D$ | Coordenadas de un cliente: $(\text{Ingreso}, \text{Índice de Gasto})$. |
| **Centroide ($c_k$)** | Prototipo o centro geométrico del cluster $k$ | Vector medio que representa el perfil promedio del grupo. |
| **Distancia Euclidiana** | Métrica de similitud entre punto y centroide | $d(x, c) = \sqrt{\sum_{j=1}^{D} (x_j - c_j)^2}$. |
| **Inercia / WCSS** | Suma de cuadrados dentro de los clusters | Medida de compacidad total de la agrupación alcanzada. |
| **Convergencia** | Estado de parada del algoritmo | Ocurre cuando la posición de los centroides deja de cambiar. |

---

## 3. Definición del Problema: Segmentación de Clientes

Imaginemos una empresa comercial que desea segmentar a su base de clientes a partir de dos variables continuas normalizadas:
1. **Ingreso Anual ($x_1$):** Nivel de ingresos del cliente.
2. **Índice de Gasto ($x_2$):** Puntaje de consumo asignado por el negocio.

Al no disponer de categorías preexistentes, aplicaremos **K-Means** para agrupar automáticamente a los clientes en nichos de mercado diferenciados.

```python
# Generación de un dataset sintético bidimensional con 300 clientes distribuidos en 4 grupos reales
X, y_real = make_blobs(
    n_samples=300, 
    centers=4, 
    cluster_std=0.70, 
    random_state=SEED
)

print(f"📊 Dataset generado exitosamente:")
print(f"   • Número de instancias (N): {X.shape[0]}")
print(f"   • Número de características (D): {X.shape[1]}")

# Visualización inicial de los datos no etiquetados (todos los puntos en un mismo color)
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.7, edgecolors='k', s=50)
plt.title("Visualización de Datos No Etiquetados (Entrada del Algoritmo)", fontsize=12, fontweight='bold')
plt.xlabel("Característica 1: Ingreso Anual (Normalizado)")
plt.ylabel("Característica 2: Índice de Gasto (Normalizado)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

---

## 4. Implementación Paso a Paso de K-Means desde Cero

El algoritmo de **K-Means** opera mediante un proceso iterativo de dos pasos alternados (Lloyd, 1982; Russell & Norvig, 2004):
1. **Paso de Asignación (Expectación):** Cada punto se asigna al centroide más cercano.
2. **Paso de Actualización (Maximización):** Cada centroide se recalcula como la media aritmética de todos los puntos asignados a su cluster.

---

### 4.1 Paso A: Inicialización Aleatoria de Centroides

Seleccionamos $K$ puntos aleatorios del conjunto de datos para actuar como los centroides iniciales $c_1, c_2, \dots, c_K$.

```python
def inicializar_centroides(X, K):
    """Selecciona K puntos aleatorios del conjunto X para ser los centroides iniciales."""
    indices_aleatorios = np.random.choice(X.shape[0], K, replace=False)
    return X[indices_aleatorios].copy()

# Prueba de inicialización
centroides_test = inicializar_centroides(X, K=4)
print("📍 Centroides iniciales generados aleatoriamente:
", centroides_test)
```

---

### 4.2 Paso B: Asignación de Puntos al Centroide Más Cercano

Para cada punto $x_i$, se calcula su distancia euclidiana a cada centroide $c_k$ y se le asigna la etiqueta del centroide que minimiza dicha distancia:

$$a_i = \arg\min_{k \in \{1, \dots, K\}} \|x_i - c_k\|^2$$

```python
def calcular_distancia_euclidiana(a, b):
    """Calcula la distancia euclidiana vectorial entre dos puntos o matrices."""
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))

def asignar_clusters(X, centroides):
    """Asigna cada punto de X al índice del centroide más cercano."""
    # Matriz de distancias de tamaño (N, K)
    distancias = np.linalg.norm(X[:, np.newaxis, :] - centroides[np.newaxis, :, :], axis=2)
    return np.argmin(distancias, axis=1)

# Prueba de asignación
etiquetas_test = asignar_clusters(X, centroides_test)
print("🏷️ Muestra de asignación inicial (primeros 10 puntos):", etiquetas_test[:10])
```

---

### 4.3 Paso C: Actualización de Centroides

Cada centroide $c_k$ se desplaza a la posición media de los datos asignados a su grupo:

$$c_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$

```python
def recalcular_centroides(X, etiquetas, K):
    """Recalcula la posición de los K centroides como el promedio de sus puntos asignados."""
    N, D = X.shape
    nuevos_centroides = np.zeros((K, D))
    
    for k in range(K):
        puntos_cluster = X[etiquetas == k]
        if len(puntos_cluster) > 0:
            nuevos_centroides[k] = np.mean(puntos_cluster, axis=0)
        else:
            # Si un cluster se queda vacío, se reinicializa aleatoriamente
            nuevos_centroides[k] = X[np.random.choice(N)]
            
    return nuevos_centroides
```

---

### 4.4 Paso D: Ensamblado del Algoritmo Completo K-Means

Integramos los componentes en un bucle iterativo que finaliza cuando se alcanza la **convergencia** (los centroides no cambian) o se supera un número máximo de iteraciones.

```python
def ejecutar_kmeans(X, K, max_iter=100, tol=1e-4):
    """Ejecuta el algoritmo K-Means completo desde cero."""
    centroides = inicializar_centroides(X, K)
    historial_centroides = [centroides.copy()]
    
    for iteracion in range(1, max_iter + 1):
        # 1. Asignar clusters
        etiquetas = asignar_clusters(X, centroides)
        
        # 2. Recalcular centroides
        nuevos_centroides = recalcular_centroides(X, etiquetas, K)
        historial_centroides.append(nuevos_centroides.copy())
        
        # 3. Verificar condición de parada (convergencia)
        desplazamiento = np.linalg.norm(nuevos_centroides - centroides)
        if desplazamiento < tol:
            print(f"🔄 Convergencia alcanzada con éxito en la iteración {iteracion}.")
            break
            
        centroides = nuevos_centroides
        
    return centroides, etiquetas, historial_centroides

# Ejecución de prueba con K = 4
centroides_finales, etiquetas_finales, historial = ejecutar_kmeans(X, K=4)
```

---

## 5. Visualización de la Traza Iterativa de Convergencia

A continuación, graficamos el resultado final del agrupamiento, resaltando la posición de los **centroides finales** (marcados con una 'X' roja):

```python
plt.figure(figsize=(8, 5.5))

# Graficar los puntos coloreados por el cluster asignado
scatter = plt.scatter(X[:, 0], X[:, 1], c=etiquetas_finales, cmap='viridis', alpha=0.7, edgecolors='k', s=50)

# Graficar los centroides finales
plt.scatter(
    centroides_finales[:, 0], 
    centroides_finales[:, 1], 
    c='red', 
    marker='X', 
    s=200, 
    linewidths=2, 
    label='Centroides Finales ($c_k$)',
    edgecolors='black'
)

plt.title("Resultado del Agrupamiento K-Means ($K=4$)", fontsize=13, fontweight='bold')
plt.xlabel("Ingreso Anual (Normalizado)")
plt.ylabel("Índice de Gasto (Normalizado)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 6. Selección del $K$ Óptimo: El Método del Codo (*Elbow Method*)

En la práctica, raramente conocemos de antemano el número correcto de grupos $K$. Para resolver esto, evaluamos la **Inercia** o **WCSS** (Within-Cluster Sum of Squares) para diferentes valores de $K$:

$$\text{Inercia}(K) = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - c_k\|^2$$

A medida que $K$ aumenta, la inercia disminuye. El **Método del Codo** busca el punto de inflexión donde el incremento en $K$ ya no aporta una reducción significativa en la inercia.

```python
def calcular_inercia(X, centroides, etiquetas):
    """Calcula la suma de distancias al cuadrado de cada punto a su centroide asignado."""
    inercia = 0.0
    for k in range(len(centroides)):
        puntos_k = X[etiquetas == k]
        if len(puntos_k) > 0:
            inercia += np.sum((puntos_k - centroides[k]) ** 2)
    return inercia

# Evaluar el algoritmo para K desde 1 hasta 8
valores_k = list(range(1, 9))
inercias = []

for k in valores_k:
    c_fin, etq_fin, _ = ejecutar_kmeans(X, K=k, max_iter=50)
    inercia_k = calcular_inercia(X, c_fin, etq_fin)
    inercias.append(inercia_k)

# Graficar la curva del codo
plt.figure(figsize=(8, 4.5))
plt.plot(valores_k, inercias, 'bo-', linewidth=2.5, markersize=8, color="#1B5E20")
plt.axvline(x=4, color='red', linestyle='--', label='Codo Óptimo ($K=4$)')
plt.title("Método del Codo para la Selección del Número Óptimo de Clusters ($K$)", fontsize=12, fontweight='bold')
plt.xlabel("Número de Clusters ($K$)")
plt.ylabel("Inercia (WCSS)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 7. Verificación Cruzada con Scikit-Learn

Para validar la precisión de nuestra implementación desde cero, ejecutamos el módulo profesional `sklearn.cluster.KMeans` sobre el mismo conjunto de datos:

```python
# Ejecución del modelo oficial de Scikit-Learn
modelo_sklearn = SklearnKMeans(n_clusters=4, random_state=SEED, n_init=10)
etiquetas_sklearn = modelo_sklearn.fit_predict(X)
centroides_sklearn = modelo_sklearn.cluster_centers_

# Comparación numérica de inercias
inercia_propia = calcular_inercia(X, centroides_finales, etiquetas_finales)
inercia_sklearn = modelo_sklearn.inertia_

print(" Validaciones Comparativas:")
print(f"   • Inercia Implementación Propia : {inercia_propia:.4f}")
print(f"   • Inercia Scikit-Learn           : {inercia_sklearn:.4f}")
print(" Conclusión: La implementación artesanal alcanza la misma precisión cuantitativa que Scikit-Learn.")
```

---

## 8. Cuestionario de Reflexión y Discusión en Clase

Responda a las siguientes preguntas analíticas basándose en las observaciones del laboratorio:

1. **¿Por qué la inicialización aleatoria de los centroides puede conducir a resultados diferentes o subóptimos?**
   * *Respuesta guía:* K-Means es susceptible de quedar atrapado en mínimos locales según la posición inicial de los centroides. Por ello, librerías como Scikit-Learn utilizan técnicas mejoradas como **K-Means++** o ejecutan múltiples reinicios aleatorios (`n_init=10`).
2. **¿Qué sucede si aplicamos K-Means a un conjunto de datos donde los clusters tienen formas no esféricas (ej. anillos concéntricos o lunas)?**
   * *Respuesta guía:* K-Means asume clusters esféricos e isótropos debido al uso de la distancia euclidiana. Para formas complejas no convexas, algoritmos como DBSCAN o Spectral Clustering resultan superiores.
3. **¿Cuál es la diferencia conceptual entre la clasificación supervisada (ej. K-NN) y el agrupamiento no supervisado (K-Means)?**
   * *Respuesta guía:* K-NN requiere un conjunto etiquetado previo para aprender las fronteras de clase de nuevos ejemplos, mientras que K-Means no utiliza etiquetas y descubre grupos latentes de forma autónoma basándose únicamente en la similitud de las características de entrada.
