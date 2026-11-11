# Práctica Guiada: Aprendizaje Automático Supervisado
## Clasificación de Afecciones Cardiacas mediante Árboles de Decisión y Evaluación Estadística

**Asignatura:** Fundamentos de Inteligencia Artificial (Unidad III)  
**Tema:** Aprendizaje Automático Supervisado, Inducción de Árboles de Decisión, Diagnóstico de Sobreajuste y Métricas de Desempeño Clínico  
**Duración:** 60 minutos  
**Entorno de ejecución:** Google Colab / Jupyter Notebook (Python 3.x)  

---

## 1. Introducción y Sustento Teórico General

El **Aprendizaje Automático Supervisado** (*Supervised Machine Learning*) es un paradigma de la Inteligencia Artificial donde un algoritmo aprende una función de mapeo $h: \mathcal{X} \to \mathcal{Y}$ a partir de un conjunto de entrenamiento etiquetado por expertos o procesos de observación previos $D = \{(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)\}$, en el cual cada instancia de entrada $x_i \in \mathcal{X}$ está asociada a una respuesta correcta $y_i \in \mathcal{Y}$ (Russell & Norvig, 2004; IBM, 2026).

En las tareas de **clasificación discreta**, la variable objetivo $y$ toma valores categóricos finitos (por ejemplo, $y \in \{0, 1\}$, donde $0 = \text{Sano}$ y $1 = \text{Afección Cardiaca Presente}$). 

El objetivo primordial del modelo no es memorizar los datos de entrenamiento, sino alcanzar una alta **capacidad de generalización** para predecir correctamente la condición médica en **pacientes nuevos no observados previamente** (Ferrante, 2020).

---

## 2. Requerimientos Técnicos e Importación de Dependencias

Para la ejecución de esta práctica se requieren las siguientes librerías estándar de procesamiento de datos, aprendizaje automático y visualización médica en Python:
- `numpy` y `pandas`: Manipulación vectorial y estructuras tabulares de datos clínicos.
- `matplotlib` y `seaborn`: Generación de gráficos estadísticos y matrices de confusión.
- `scikit-learn`: Ajuste de modelos de clasificación, división de conjuntos y cálculo de métricas de desempeño.

Ejecute la siguiente celda para inicializar el entorno e importar los módulos requeridos:

```python
# Carga e importación de dependencias
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    classification_report, 
    roc_auc_score
)

# Fijar semilla aleatoria para garantizar reproducibilidad exacta en laboratorio
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("✅ Entorno configurado correctamente. Semilla fijada en SEED =", SEED)
```

---

## 3. Paso 1: Configuración del Espacio de Características y Carga del Dataset Clínico

### Sustento Teórico:
El espacio de características $\mathcal{X} \subset \mathbb{R}^d$ representa el conjunto de atributos clínicos cuantitativos y cualitativos extraídos del paciente (edad, presión arterial, colesterol, frecuencia cardiaca, etc.). Cada paciente es representado como un vector $x = (x^{(1)}, x^{(2)}, \dots, x^{(d)})$. La etiqueta $y \in \{0, 1\}$ actúa como la supervisión humana (diagnóstico confirmado mediante angiografía u otros estudios avanzados).

```python
# Paso 1: Simulación estructurada de un dataset clínico de afecciones cardiacas (300 pacientes)
def generar_dataset_cardiaco(n_muestras=300, random_state=42):
    np.random.seed(random_state)
    
    # Generar características clínicas realistas
    edad = np.random.randint(30, 75, size=n_muestras)
    presion_sistolica = np.random.normal(130, 18, size=n_muestras).clip(90, 200)
    colesterol = np.random.normal(240, 45, size=n_muestras).clip(130, 400)
    frecuencia_max = np.random.normal(150, 22, size=n_muestras).clip(80, 202)
    angina_ejercicio = np.random.choice([0, 1], size=n_muestras, p=[0.65, 0.35])
    
    # Modelo generador subyacente con riesgo ponderado + ruido estocástico
    score_riesgo = (
        0.04 * (edad - 50) + 
        0.03 * (presion_sistolica - 120) + 
        0.02 * (colesterol - 200) - 
        0.03 * (frecuencia_max - 140) + 
        1.5 * angina_ejercicio + 
        np.random.normal(0, 1.0, size=n_muestras)
    )
    
    # Etiqueta binaria (1: Diagnóstico de Afección Cardiaca, 0: Sano)
    diagnostico = (score_riesgo > 0.5).astype(int)
    
    df = pd.DataFrame({
        'Edad': edad,
        'PresionSistolica': np.round(presion_sistolica, 1),
        'Colesterol': np.round(colesterol, 1),
        'FrecuenciaMax': np.round(frecuencia_max, 1),
        'AnginaEjercicio': angina_ejercicio,
        'DiagnosticoCardiopatia': diagnostico
    })
    return df

# Carga y vista previa
df_cardio = generar_dataset_cardiaco(n_muestras=300, random_state=SEED)
print(f"Dataset generado exitosamente: {df_cardio.shape[0]} pacientes, {df_cardio.shape[1]-1} características.")
df_cardio.head()
```

---

## 4. Paso 2: División Estratificada en Entrenador y Prueba (Train/Test Split)

### Sustento Teórico:
Para evitar el fenómeno de **contaminación del conjunto de prueba** (*peeking*) y medir la verdadera capacidad de generalización del modelo sobre datos nunca vistos, el conjunto de datos completo $D$ se particiona en dos subconjuntos disjuntos: **Entrenamiento ($D_{train}$, 80%)** y **Prueba ($D_{test}$, 20%)** (Russell & Norvig, 2004). Se utiliza la **estratificación** para mantener la misma proporción de clases en ambas particiones.

```python
# Paso 2: Separar características (X) de la variable objetivo (y) y dividir los conjuntos
X = df_cardio.drop(columns=['DiagnosticoCardiopatia'])
y = df_cardio['DiagnosticoCardiopatia']

# División 80% Entrenamiento y 20% Prueba con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=SEED, 
    stratify=y
)

print(f"Muestras de Entrenamiento: {X_train.shape[0]} pacientes | Muestras de Prueba: {X_test.shape[0]} pacientes")
print("Distribución de clases en entrenamiento (% Positivos):", f"{y_train.mean()*100:.1f}%")
```

---

## 5. Paso 3: Inducción de Árboles de Decisión (Entropía y Ganancia de Información)

### Sustento Teórico:
Los **Árboles de Decisión** (algoritmos ID3 / C4.5 / CART) organizan el conocimiento mediante pruebas de división jerárquica en los atributos. Para seleccionar la mejor división en cada nodo, se calcula la **Entropía de Shannon** $I(D)$, la cual mide el grado de impureza o desorden de las clases en la muestra:

$$I(D) = - \sum_{c \in \{0, 1\}} p_c \log_2(p_c)$$

La elección del atributo y el punto de corte óptimo se basa en la **Ganancia de Información** $\text{Gain}(A)$, buscando maximizar la reducción del desorden (Russell & Norvig, 2004; Luger, 2004):

$$\text{Gain}(A) = I(D) - \sum_{v \in \text{Valores}(A)} \frac{|D_v|}{|D|} I(D_v)$$

```python
# Paso 3: Instanciar y entrenar un Árbol de Decisión sin restricciones iniciales (Ilimitado)
clf_arbol_completo = DecisionTreeClassifier(
    criterion='entropy', 
    random_state=SEED
)

# Entrenar el modelo induciendo las reglas a partir de D_train
clf_arbol_completo.fit(X_train, y_train)

# Evaluación preliminar de exactitud (Accuracy)
acc_train_comp = clf_arbol_completo.score(X_train, y_train)
acc_test_comp = clf_arbol_completo.score(X_test, y_test)

print(f"Exactitud en Entrenamiento (Árbol Completo): {acc_train_comp * 100:.2f}%")
print(f"Exactitud en Prueba (Árbol Completo)        : {acc_test_comp * 100:.2f}%")
```

---

## 6. Paso 4: Visualización del Árbol e Interpretación de Reglas Clínicas

### Sustento Teórico:
Una de las mayores ventajas de los árboles de decisión en medicina es su **interpretabilidad y transparencia** (*modelos simbólicos de caja blanca*), permitiendo a los médicos auditar las reglas lógicas antes de utilizarlas en la toma de decisiones clínicas.

```python
# Paso 4: Graficar la estructura jerárquica del árbol inducido
plt.figure(figsize=(16, 8))
plot_tree(
    clf_arbol_completo,
    feature_names=X.columns,
    class_names=['Sano', 'Cardiopatía'],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Estructura Jerárquica del Árbol de Decisión Ilimitado", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## 7. Paso 5: Diagnóstico de Sobreajuste (*Overfitting*) y Poda del Árbol (*Pruning*)

### Sustento Teórico:
Si el árbol crece de forma ilimitada hasta separar cada muestra individual, memorizará el ruido de entrenamiento, generando una **varianza excesiva y sobreajuste (*overfitting*)**. Aplicando la **Navaja de Ockham**, debemos preferir la hipótesis consistente más sencilla (Russell & Norvig, 2004). Para ello se aplica la **poda pre-entrenamiento** fijando la profundidad máxima (`max_depth`) y el número mínimo de muestras por hoja (`min_samples_leaf`).

```python
# Paso 5: Construir un Árbol Podado (Regularizado) para mejorar la generalización
clf_arbol_podado = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,                 # Restricción de profundidad máxima
    min_samples_leaf=5,          # Mínimo de pacientes requeridos por hoja
    random_state=SEED
)

clf_arbol_podado.fit(X_train, y_train)

acc_train_poda = clf_arbol_podado.score(X_train, y_train)
acc_test_poda = clf_arbol_podado.score(X_test, y_test)

print("=== COMPARATIVA DE RENDIMIENTO: SOBREAJUSTE VS. REGULARIZACIÓN ===")
print(f"Árbol Completo -> Train Accuracy: {acc_train_comp*100:.1f}% | Test Accuracy: {acc_test_comp*100:.1f}% (Brecha: {(acc_train_comp-acc_test_comp)*100:.1f}%)")
print(f"Árbol Podado   -> Train Accuracy: {acc_train_poda*100:.1f}% | Test Accuracy: {acc_test_poda*100:.1f}% (Brecha: {(acc_train_poda-acc_test_poda)*100:.1f}%)")

# Visualizar el árbol podado e interpretable
plt.figure(figsize=(12, 6))
plot_tree(
    clf_arbol_podado,
    feature_names=X.columns,
    class_names=['Sano', 'Cardiopatía'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Árbol de Decisión Podado (max_depth=3) - Modelo Simplificado", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## 8. Paso 6: Evaluación Estadística con Matriz de Confusión y Métricas Clínicas

### Sustento Teórico:
En el ámbito médico, la exactitud (*accuracy*) global no es suficiente. La **Matriz de Confusión** desglosa los resultados en:
- **Verdaderos Positivos ($VP$):** Pacientes enfermos identificados correctamente.
- **Falsos Positivos ($FP$):** Pacientes sanos diagnosticados erróneamente como enfermos.
- **Verdaderos Negativos ($VN$):** Pacientes sanos identificados correctamente.
- **Falsos Negativos ($FN$):** Pacientes enfermos no detectados por el modelo (**riesgo clínico crítico**).

Métricas clave:
$$\text{Precisión} = \frac{VP}{VP + FP}, \quad \text{Sensibilidad (Recall)} = \frac{VP}{VP + FN}, \quad F_1\text{-Score} = 2 \times \frac{\text{Precisión} \times \text{Sensibilidad}}{\text{Precisión} + \text{Sensibilidad}}$$

```python
# Paso 6: Evaluación del modelo podado en el conjunto de prueba D_test
y_pred_poda = clf_arbol_podado.predict(X_test)
y_prob_poda = clf_arbol_podado.predict_proba(X_test)[:, 1]

# 1. Matriz de Confusión
cm = confusion_matrix(y_test, y_pred_poda)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Sano', 'Cardiopatía'])

plt.figure(figsize=(6, 5))
disp.plot(cmap='Greens', values_format='d')
plt.title("Matriz de Confusión en Prueba (Diagnóstico Cardiaco)", fontsize=11, fontweight='bold')
plt.show()

# 2. Reporte de Clasificación Detallado
print("=== REPORTE DE CLASIFICACIÓN ESTADÍSTICA ===")
print(classification_report(y_test, y_pred_poda, target_names=['Sano (0)', 'Cardiopatía (1)']))

# 3. Área Bajo la Curva ROC (ROC-AUC)
auc_score = roc_auc_score(y_test, y_prob_poda)
print(f"Puntuación ROC-AUC: {auc_score:.4f}")
```

---

## 9. Paso 7: Validación Cruzada de K-Pliegues (K-Fold Cross-Validation)

### Sustento Teórico:
Para obtener una estimación insesgada del rendimiento del algoritmo sin depender de una única división aleatoria de los datos, se aplica **Validación Cruzada de $K$ pasadas ($K$-Fold Cross-Validation)** con $K=5$, promediando el rendimiento a lo largo de 5 particiones disjuntas (Russell & Norvig, 2004).

```python
# Paso 7: Validación Cruzada de 5 pliegues sobre la totalidad de las muestras
scores_cv = cross_val_score(clf_arbol_podado, X, y, cv=5, scoring='f1')

print("=== VALIDACIÓN CRUZADA DE 5 PLIEGUES (MÉTRICA F1-SCORE) ===")
for fold, score in enumerate(scores_cv, 1):
    print(f"  Pliegue {fold}: F1-Score = {score:.4f}")

print(f"
📢 Rendimiento Promedio F1-Score: {scores_cv.mean():.4f} (+/- {scores_cv.std()*2:.4f})")
```

---

## 10. Cuestionario de Reflexión y Discusión en Clase

Responda a las siguientes preguntas a partir de los resultados obtenidos en la práctica:

1. **¿Por qué un Falso Negativo ($FN$) es significativamente más peligroso que un Falso Positivo ($FP$) en la clasificación de cardiopatías?**
   * *Respuesta guía:* Un Falso Negativo implica enviar a casa a un paciente enfermo sin tratamiento médico, con riesgo de infarto u otro evento severo. Un Falso Positivo implica realizar pruebas adicionales confirmatorias a un paciente sano, lo cual genera costos o ansiedad, pero no arriesga su vida de inmediato. Por ello, la **Sensibilidad (*Recall*)** es prioritaria.
2. **¿Cómo se evidencia empíricamente el sobreajuste en el Árbol Completo frente al Árbol Podado?**
   * *Respuesta guía:* El árbol completo alcanza un 100% de exactitud en entrenamiento pero cae drásticamente en el conjunto de prueba. El árbol podado reduce la exactitud de entrenamiento pero mantiene un rendimiento consistente en prueba, demostrando mejor generalización.
3. **¿Cuál es el rol de la Entropía y la Ganancia de Información en la selección del atributo raíz?**
   * *Respuesta guía:* La Entropía mide la impureza. El algoritmo evalúa qué característica (ej. `AnginaEjercicio` o `FrecuenciaMax`) divide mejor a la población para maximizar la reducción del desorden (Ganancia de Información), colocándola en la raíz del árbol.
