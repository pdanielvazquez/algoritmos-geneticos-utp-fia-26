# Práctica Guiada: Algoritmo Genético - "Adivina la Frase Objetivo"

**Asignatura:** Fundamentos de Inteligencia Artificial (Unidad II)  
**Tema:** Algoritmos Evolutivos y Optimización Poblacional  
**Duración:** 60 minutos  
**Entorno de ejecución:** Google Colab / Jupyter Notebook (Python 3.x)  

---

## 1. Requerimientos Técnicos e Instalación de Dependencias

Para la ejecución de esta práctica se requieren las siguientes librerías estándar de procesamiento numérico y visualización gráfica:
- `numpy`: Para manipulación eficiente de vectores y operaciones numéricas.
- `matplotlib`: Para la generación de gráficos de convergencia evolutiva.
- `tqdm`: Para mostrar barras de progreso interactivas durante el ciclo evolutivo.

Ejecute la siguiente celda de código para asegurar la presencia de las dependencias e importar los módulos requeridos:

```python
# Instalación de librerías necesarias (descomentar si ejecuta en Google Colab o entorno nuevo)
# !pip install numpy matplotlib tqdm

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Fijar semilla aleatoria para garantizar la reproducibilidad de los resultados en el laboratorio
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("✅ Entorno configurado correctamente. Semilla fijada en SEED =", SEED)
```

---

## 2. Fundamentos Teóricos e Interpretación Biológica

Los **Algoritmos Genéticos (AGs)** son métodos de búsqueda heurística y optimización inspirados en los mecanismos de la **evolución natural** y la **selección darwiniana** (Darwin, 1859; Holland, 1975). A diferencia de los métodos de búsqueda local de un solo estado (como *Hill Climbing*), los AGs mantienen y evolucionan simultáneamente una **población** de posibles soluciones (*individuos*).

### Mapeo de Conceptos: Biología vs. Computación

| Concepto Biológico | Equivalente en Algoritmos Genéticos | Definición en el Ejercicio |
| :--- | :--- | :--- |
| **Cromosoma / Individuo** | Solución candidata completa | Una cadena de caracteres de longitud $N$ (ej. `"IA EVOLUTIVA 2026"`). |
| **Gen** | Variable o componente individual | La posición o carácter en un índice específico de la cadena. |
| **Alelo** | Valor concreto asignado a un gen | El carácter específico asignado (ej. `'A'`, `'B'`, `'5'`, `' '`). |
| **Población** | Conjunto de soluciones en una generación | Matriz o lista de $M$ cadenas de caracteres. |
| **Aptitud (*Fitness*)** | Grado de adaptación al entorno | Número de caracteres que coinciden exactamente con la frase objetivo. |
| **Selección** | Supervivencia del más apto | Elección estocástica de progenitores con probabilidad proporcional a su *fitness*. |
| **Cruce (*Crossover*)** | Reproducción sexual y recombinación | Mezcla de segmentos de dos cadenas padres en un punto de corte. |
| **Mutación** | Alteración genética estocástica | Cambio aleatorio de un carácter por otro del alfabeto con probabilidad $p_m$. |

---

## 3. Definición del Problema y Espacio de Búsqueda

En el problema de **Adivinanza de Frase Objetivo** (*Target String Matching*), el objetivo es evolucionar una población de cadenas aleatorias hasta reconstruir exactamente una cadena meta definida.

### Formulación Matemática:
- **Alfabeto ($\mathcal{A}$):** Letras mayúsculas de la 'A' a la 'Z', números del '0' al '9' y el espacio en blanco `' '` (Total: 37 caracteres).
- **Frase Objetivo:** `"IA EVOLUTIVA 2026"` (Longitud $N = 17$).
- **Tamaño del Espacio de Estados:** $|\mathcal{A}|^N = 37^{17} pprox 4.29 	imes 10^{26}$ combinaciones posibles.

> **¿Por qué utilizar un Algoritmo Genético?**  
> Probar $4.29 	imes 10^{26}$ combinaciones mediante búsqueda a ciegas (*fuerza bruta*) a una velocidad de 1,000,000 de evaluaciones por segundo tomaría aproximadamente **$13.6$ mil millones de años** (la edad del universo). Un Algoritmo Genético encuentra la solución exacta en **menos de 3 segundos** (habitualmente en menos de 100 generaciones).

```python
# Configuración del espacio de búsqueda y parámetros globales
ALFABETO = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ0123456789 "
FRASE_OBJETIVO = "IA EVOLUTIVA 2026"
LONGITUD_FRASE = len(FRASE_OBJETIVO)
TAMANO_ALFABETO = len(ALFABETO)

espacio_estados = TAMANO_ALFABETO ** LONGITUD_FRASE
print(f"🎯 Frase Objetivo    : '{FRASE_OBJETIVO}'")
print(f"📏 Longitud (N)      : {LONGITUD_FRASE} caracteres")
print(f"🔤 Tamaño Alfabeto   : {TAMANO_ALFABETO} caracteres")
print(f"🌌 Espacio de Estados: {espacio_estados:.2e} combinaciones posibles")
```

---

## 4. Implementación Paso a Paso de los Componentes

### 4.1 Inicialización de la Población

Cada individuo se genera como una cadena de $N$ caracteres seleccionados aleatoriamente del alfabeto de forma uniforme.

```python
def crear_individuo(longitud):
    """Genera una cadena aleatoria de longitud N basada en el alfabeto."""
    return ''.join(random.choice(ALFABETO) for _ in range(longitud))

def crear_poblacion(tamano, longitud):
    """Crea una lista de 'tamano' individuos aleatorios."""
    return [crear_individuo(longitud) for _ in range(tamano)]

# Prueba de generación
poblacion_inicial_ejemplo = crear_poblacion(5, LONGITUD_FRASE)
print("Muestra de población inicial aleatoria:")
for i, ind in enumerate(poblacion_inicial_ejemplo, 1):
    print(f"  Individuo {i}: '{ind}'")
```

---

### 4.2 Función de Aptitud (*Fitness Function*)

La **función de aptitud** mide qué tan "cerca" está una solución de la meta. En este caso, cuenta el número de caracteres que coinciden exactamente en posición con la frase objetivo:

$$	ext{Fitness}(x) = \sum_{i=1}^{N} \mathbb{I}(x[i] == 	ext{objetivo}[i])$$

Donde $\mathbb{I}$ es la función indicadora (retorna 1 si coinciden, 0 si no). La aptitud máxima perfecta es $N = 17$.

```python
def evaluar_fitness(individuo, objetivo):
    """Calcula el número de caracteres que coinciden exactamente con el objetivo."""
    return sum(1 for a, b in zip(individuo, objetivo) if a == b)

# Prueba de evaluación
ejemplo_ind = "IA EVOLUXIVA 2020"
fit_val = evaluar_fitness(ejemplo_ind, FRASE_OBJETIVO)
print(f"Evaluación del individuo '{ejemplo_ind}': {fit_val}/{LONGITUD_FRASE} aciertos.")
```

---

### 4.3 Operador de Selección: Selección por Torneo

En la **selección por torneo**, se eligen al azar $k$ individuos de la población y el que posee la mayor aptitud gana el derecho a ser progenitor. Un valor de $k=3$ ofrece un excelente balance entre la *presión de selección* y la *preservación de la diversidad*.

```python
def seleccion_torneo(poblacion, fitnesses, k=3):
    """Selecciona al mejor individuo de un subconjunto aleatorio de tamaño k."""
    seleccionados_indices = random.sample(range(len(poblacion)), k)
    mejor_idx = max(seleccionados_indices, key=lambda idx: fitnesses[idx])
    return poblacion[mejor_idx]
```

---

### 4.4 Operador de Cruce (*Crossover*): Cruce de un Punto

El **cruce de un punto** selecciona un índice de corte aleatorio $1 \le p < N$. Los dos hijos se forman combinando la primera parte de un padre con la segunda parte del otro. Se ejecuta con una probabilidad de cruce $p_c pprox 0.85$.

```python
def cruce_un_punto(padre1, padre2, prob_cruce=0.85):
    """Combina dos padres en un punto de corte aleatorio con probabilidad prob_cruce."""
    if random.random() < prob_cruce:
        punto_corte = random.randint(1, len(padre1) - 1)
        hijo1 = padre1[:punto_corte] + padre2[punto_corte:]
        hijo2 = padre2[:punto_corte] + padre1[punto_corte:]
        return hijo1, hijo2
    return padre1, padre2

# Prueba de cruce
p1, p2 = "AAAAAFFFFFF2026", "IA EVOLUTIVABBBB"
h1, h2 = cruce_un_punto(p1, p2, prob_cruce=1.0)
print(f"Padre 1: '{p1}'
Padre 2: '{p2}'
Hijo 1 : '{h1}'
Hijo 2 : '{h2}'")
```

---

### 4.5 Operador de Mutación Puntual

La **mutación** recorre cada carácter del individuo y, con una probabilidad muy baja $p_m pprox 0.01$ (1%), lo reemplaza por un nuevo carácter aleatorio del alfabeto. La mutación es fundamental para introducir alelos perdidos y evitar el estancamiento en máximos locales.

```python
def mutacion(individuo, prob_mutacion=0.01):
    """Altera aleatoriamente caracteres individuales con probabilidad prob_mutacion."""
    resultado = []
    for char in individuo:
        if random.random() < prob_mutacion:
            resultado.append(random.choice(ALFABETO))
        else:
            resultado.append(char)
    return ''.join(resultado)
```

---

## 5. Bucle Evolutivo Principal con Elitismo

El algoritmo integra la estrategia de **elitismo**, preservando intacto al mejor individuo de la generación previa en la nueva población. Esto garantiza una propiedad de convergencia monótona no decreciente ($	ext{Fitness}_{	ext{máx}}(t+1) \ge 	ext{Fitness}_{	ext{máx}}(t)$).

```python
def algoritmo_genetico(objetivo, tam_poblacion=150, max_generaciones=200, 
                       prob_cruce=0.85, prob_mutacion=0.01, k_torneo=3, verbose=True):
    """Ejecuta el ciclo de vida completo del Algoritmo Genético."""
    longitud = len(objetivo)
    poblacion = crear_poblacion(tam_poblacion, longitud)
    
    historial_max = []
    historial_avg = []
    
    for gen in range(1, max_generaciones + 1):
        # 1. Evaluación de la población
        fitnesses = [evaluar_fitness(ind, objetivo) for ind in poblacion]
        
        max_fit = max(fitnesses)
        avg_fit = sum(fitnesses) / tam_poblacion
        mejor_ind = poblacion[fitnesses.index(max_fit)]
        
        historial_max.append(max_fit)
        historial_avg.append(avg_fit)
        
        if verbose and (gen == 1 or gen % 10 == 0 or max_fit == longitud):
            print(f"Gen {gen:03d} | Fit Máx: {max_fit:02d}/{longitud} | Fit Prom: {avg_fit:5.2f} | Mejor: '{mejor_ind}'")
            
        # Criterio de parada: ¿Alcanzó la meta?
        if max_fit == longitud:
            if verbose:
                print(f"
🎉 ¡SOLUCIÓN ENCONTRADA EN LA GENERACIÓN {gen}!")
                print(f"Resultados Finales: '{mejor_ind}' (Aptitud: {max_fit}/{longitud})")
            break
            
        # 2. Elitismo: Mantener al mejor individuo sin modificaciones
        nueva_poblacion = [mejor_ind]
        
        # 3. Reproducción hasta completar la nueva población
        while len(nueva_poblacion) < tam_poblacion:
            padre1 = seleccion_torneo(poblacion, fitnesses, k_torneo)
            padre2 = seleccion_torneo(poblacion, fitnesses, k_torneo)
            
            hijo1, hijo2 = cruce_un_punto(padre1, padre2, prob_cruce)
            
            nueva_poblacion.append(mutacion(hijo1, prob_mutacion))
            if len(nueva_poblacion) < tam_poblacion:
                nueva_poblacion.append(mutacion(hijo2, prob_mutacion))
                
        poblacion = nueva_poblacion
        
    return historial_max, historial_avg, mejor_ind

# Ejecución principal
print("🚀 Iniciando la ejecución del Algoritmo Genético...
")
hist_max, hist_avg, solucion = algoritmo_genetico(
    objetivo=FRASE_OBJETIVO,
    tam_poblacion=150,
    max_generaciones=200,
    prob_cruce=0.85,
    prob_mutacion=0.01,
    k_torneo=3
)
```

---

## 6. Visualización Gráfica de Convergencia

Ejecute la siguiente celda para generar el gráfico de la curva de aprendizaje del algoritmo, comparando la evolución del **Fitness Máximo** frente al **Fitness Promedio**:

```python
plt.figure(figsize=(10, 5))
plt.plot(hist_max, label="Aptitud Máxima (Elitismo)", color="#1B5E20", linewidth=2.5)
plt.plot(hist_avg, label="Aptitud Promedio de la Población", color="#66BB6A", linestyle="--", linewidth=2)
plt.axhline(y=LONGITUD_FRASE, color="red", linestyle=":", label=f"Óptimo Global ({LONGITUD_FRASE})")

plt.title("Curva de Convergencia del Algoritmo Genético", fontsize=14, fontweight="bold")
plt.xlabel("Generación", fontsize=11)
plt.ylabel("Aptitud (Caracteres Correctos)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
```

---

## 7. Experimento de Sensibilidad: El Rol de la Mutación (10 min)

La mutación es el motor de exploración. A continuación, comparemos qué sucede cuando la tasa de mutación cambia de un valor óptimo ($p_m = 0.01$) a dos escenarios extremos:
1. **Sin Mutación ($p_m = 0.00$):** Búsqueda basada únicamente en la combinación de alelos iniciales (*convergencia prematura*).
2. **Mutación Excesiva ($p_m = 0.30$):** Comportamiento casi aleatorio que destruye las buenas soluciones (*deriva o ruido genético*).

```python
tasas_mutacion = [0.00, 0.01, 0.30]
resultados_exp = {}

for pm in tasas_mutacion:
    random.seed(SEED)
    np.random.seed(SEED)
    h_max, _, _ = algoritmo_genetico(
        objetivo=FRASE_OBJETIVO,
        tam_poblacion=150,
        max_generaciones=150,
        prob_cruce=0.85,
        prob_mutacion=pm,
        verbose=False
    )
    resultados_exp[f"p_m = {pm}"] = h_max

plt.figure(figsize=(10, 5))
plt.plot(resultados_exp["p_m = 0.0"], label="p_m = 0.00 (Sin Mutación -> Estancamiento)", color="red", linestyle="--")
plt.plot(resultados_exp["p_m = 0.01"], label="p_m = 0.01 (Tasa Óptima -> Convergencia)", color="green", linewidth=2.5)
plt.plot(resultados_exp["p_m = 0.3"], label="p_m = 0.30 (Mutación Excesiva -> Ruido)", color="orange", linestyle=":")
plt.axhline(y=LONGITUD_FRASE, color="black", linestyle=":", label="Meta")

plt.title("Estudio de Sensibilidad: Impacto de la Tasa de Mutación (p_m)", fontsize=13, fontweight="bold")
plt.xlabel("Generación")
plt.ylabel("Aptitud Máxima")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
```

---

## 8. Cuestionario de Reflexión y Discusión en Clase

Responda brevemente a las siguientes preguntas a partir de los resultados observados:

1. **¿Por qué el escenario sin mutación ($p_m = 0.0$) se estanca antes de alcanzar el objetivo de 17 aciertos?**
   * *Respuesta guía:* Porque si un carácter del alfabeto (alelo) no estuvo presente en ninguna cadena de la población inicial en la posición correcta, el operador de cruce por sí solo no puede crearlo. La mutación es indispensable para recuperar diversidad perdida.
2. **¿Qué sucede cuando la tasa de mutación es demasiado alta ($p_m = 0.30$)?**
   * *Respuesta guía:* La búsqueda se convierte en un paseo aleatorio. Los bloques de construcción adaptativos obtenidos por el cruce se destruyen constantemente antes de poder consolidarse en la siguiente generación.
3. **¿Cuál es la función del Elitismo en este algoritmo?**
   * *Respuesta guía:* Previene la regresión genética, asegurando que la aptitud máxima nunca decrezca de una generación a la otra.
