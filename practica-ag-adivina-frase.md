# Práctica Guiada: Algoritmo Genético - "Adivina la Frase Objetivo"

**Asignatura:** Fundamentos de Inteligencia Artificial (Unidad II)  
**Tema:** Algoritmos Genéticos y Computación Evolutiva  
**Duración estimada:** 60 minutos  
**Entorno de ejecución:** Python 3 / Google Colab / Jupyter Notebooks  

---

## 1. Requerimientos Técnicos e Instalación de Dependencias

Para ejecutar esta práctica en **Google Colab** o en un entorno local de **Jupyter Notebook**, asegúrate de contar con las librerías científicas e instructivas básicas (, , ).





---

## 2. Fundamentos Teóricos e Interpretación Biológica

Los **Algoritmos Genéticos (AGs)** son métodos de búsqueda heurística y optimización inspirados en la teoría de la evolución natural de Charles Darwin (1859) y formalizados por John Holland (1975).

### Analogía entre la Biología y la Computación Evolutiva

| Concepto Biológico | Equivalente en Computación Evolutiva | Implementación en este Ejercicio |
| :--- | :--- | :--- |
| **Organismo / Individuo** | Solución candidata del problema | Cadena de caracteres (ej. ) |
| **Cromosoma / Genotipo** | Estructura de datos que codifica la solución | Arreglo/String de longitud $ |
| **Gen** | Unidad elemental de información / posición | Carácter individual en una posición específica |
| **Alelo** | Valor concreto asignado a un gen | Carácter del alfabeto (letra, número o espacio) |
| **Adaptación / Fitness** | Capacidad de supervivencia en el entorno | Función que cuenta coincidencias exactas con el objetivo |
| **Población** | Conjunto de individuos en una generación | Lista de $ cadenas de caracteres |
| **Selección Natural** | Reproducción diferencial de los más aptos | Selección por Torneo (=3$) |
| **Cruce (Crossover)** | Recombinación del material genético de los padres | Cruce de un punto sobre las cadenas |
| **Mutación** | Alteración aleatoria en la secuencia de ADN | Cambio aleatorio de un carácter con probabilidad $ |

---

## 3. Definición del Problema y Espacio de Búsqueda

En esta práctica resolveremos el problema de **Target String Matching** (Adivina la Frase Objetivo).

* **Frase Objetivo ($):** 
* **Longitud ($):** 17 caracteres.
* **Alfabeto ($):** Letras mayúsculas (A-Z, Ñ), dígitos (0-9) y espacio en blanco (37 caracteres en total).

### Tamaño del Espacio de Búsqueda
Dado que cada posición de la cadena de longitud =17$ puede tomar cualquiera de los 37 caracteres del alfabeto, el número total de soluciones posibles es:

10|S| = |A|^N = 37^{17} pprox 4.3 	imes 10^{26} 	ext{ soluciones posibles}10

Una búsqueda a ciegas o por fuerza bruta tardaría miles de años en encontrar la frase exacta. El Algoritmo Genético usará la **función de aptitud** como una brújula orientadora sobre el *fitness landscape* para converger en pocas generaciones.

---

## 4. Implementación Paso a Paso

### 4.1. Configuración de Parámetros Globales



### 4.2. Inicialización de la Población

Cada individuo se genera seleccionando caracteres aleatorios del alfabeto con distribución uniforme.



### 4.3. Función de Aptitud (*Fitness Function*)

La función de aptitud evalúa qué tan "adaptado" está un individuo contando el número de caracteres que coinciden exactamente en posición con la frase objetivo.

10	ext{Fitness}(x) = \sum_{i=1}^{N} \mathbb{I}(x[i] == T[i])10



### 4.4. Operador de Selección: Torneo Competitivo

En la **Selección por Torneo**, se eligen al azar $ individuos de la población y el que posee la mayor aptitud gana el derecho a ser progenitor.



### 4.5. Operador de Cruce (*Crossover*)

El **cruce de un punto** selecciona un punto de corte aleatorio en la cadena y combina la primera parte del Padre 1 con la segunda parte del Padre 2 para producir dos hijos.



### 4.6. Operador de Mutación

La mutación altera aleatoriamente cada gen (carácter) con una probabilidad muy baja $. Mantiene la diversidad genética y evita que el algoritmo quede atrapado en mínimos o máximos locales.



---

## 5. Algoritmo Genético Completo con Elitismo

El **Elitismo** garantiza que el mejor individuo de la generación actual pase intacto a la siguiente generación, impidiendo que los operadores estocásticos destruyan la mejor solución encontrada hasta el momento.



---

## 6. Visualización de la Curva de Convergencia

Graficar la aptitud máxima y promedio a través de las generaciones permite analizar la velocidad de convergencia y la diversidad de la población.



---

## 7. Experimentos de Aula: Estudio del Impacto de la Mutación (10 min)

Prueba cómo varía el desempeño del algoritmo modificando la tasa de mutación $.



---

## 8. Cuestionario de Reflexión para el Alumno

1. **¿Qué sucede cuando la tasa de mutación es  = 0.02**  
   *Respuesta esperada:* El algoritmo pierde la diversidad genética inicial. Si un carácter correcto no estaba presente en ninguna posición de la población inicial, el cruce por sí solo no podrá generarlo, provocando un estancamiento en un subóptimo local.

2. **¿Qué ocurre cuando la tasa de mutación es excesivamente alta ( = 0.35$)?**  
   *Respuesta esperada:* El algoritmo se comporta como una búsqueda aleatoria pura. La alta probabilidad de mutación destruye continuamente las soluciones buenas heredadas por el cruce, impidiendo la convergencia estable hacia la frase objetivo.

3. **¿Cuál es el rol del Elitismo en la curva de convergencia?**  
   *Respuesta esperada:* Garantizar la monotonía no decreciente de la aptitud máxima. Asegura que el progreso logrado nunca se pierda por destructividad de las operaciones estocásticas de cruce o mutación.
