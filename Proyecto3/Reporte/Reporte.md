# Informe de Proyecto: La Generación Z y la Crisis de Sentido en la Era Digital

**Institución:** Instituto Tecnológico de Morelia
**Carrera:** Ingeniería en Sistemas Computacionales
**Estudiante:** Manuel Arenazas Gámez
**Asignatura:** Inteligencia Artificial (Proyecto 3)
**Periodo:** Agosto - Diciembre 2025

---

## 1. Introducción

La presente propuesta tiene como objetivo realizar un análisis de datos mediante técnicas RAG (*Retrieval-Augmented Generation*) para explorar dos grandes problemáticas filosóficas contemporáneas: la posible crisis de sentido en la Generación Z debido a la hiperconectividad y la pérdida gradual de autonomía humana frente al avance de los algoritmos y la inteligencia artificial.

Este proyecto integra análisis cualitativo y cuantitativo, minería de texto y modelos generativos para comprender cómo los algoritmos moldean el sentido de vida, la identidad y la autonomía en la juventud actual.

## 2. Metodología

Para la ejecución de este análisis se implementó un sistema de **Generación Aumentada por Recuperación (RAG)** con las siguientes características técnicas y teóricas:

* **Motor de IA:** Modelo Llama 3.2 ejecutado localmente vía Ollama.
* **Plataforma de Orquestación:** AnythingLLM Desktop.
* **Base de Datos Vectorial:** LanceDB para el almacenamiento de embeddings.
* **Corpus de Datos:**
    * *Dataset Empírico:* Archivo `dataset_sintetico_5000_ampliado.csv` conteniendo tweets y expresiones de usuarios de la Generación Z.
    * *Contexto Teórico:* Archivo `contextos_filosoficos.csv` con definiciones operativas de Sartre, Camus, Lyotard, Bauman, Han, Foucault, Heidegger y Habermas.

El sistema fue configurado con un *System Prompt* específico para cruzar la data empírica con los marcos filosóficos, obligando al modelo a interpretar los patrones de lenguaje bajo lentes teóricos (ej. interpretar "ansiedad por likes" bajo la "Sociedad del Cansancio").

---

## 3. Resultados del Análisis de Datos

A continuación se presentan los hallazgos obtenidos tras interrogar al sistema RAG, divididos por los ejes temáticos del proyecto.

### 3.1. Eje: Vacío Existencial y Sentido (Sartre, Camus, Lyotard)
El análisis detectó que la Gen Z no utiliza terminología filosófica explícita, pero sí expresiones que denotan la "Náusea" sartreana y el "Absurdo" de Camus:
* **Terminología:** Se identificaron términos recurrentes como "vacío interior", "ausencia emocional" y "expectativas irreales".
* **Patrones de Crisis:** El lenguaje utilizado incluye metáforas de desorientación ("laberinto sin salida", "río sin destino") y una predominancia de cuestionamientos sobre la realidad y el propósito ("¿por qué estoy aquí?").
* **Rechazo a Metarrelatos:** Se confirmó la hipótesis de Lyotard. Existe un rechazo explícito a la autoridad tradicional, la religión institucional y la idea de "éxito" social convencional, en favor de una búsqueda de autenticidad individual fragmentada.

### 3.2. Eje: Identidad Líquida y Algoritmos (Bauman)
Los datos confirman la teoría de la **Modernidad Líquida**:
* **Construcción del Yo:** Los usuarios perciben que los algoritmos no solo recomiendan, sino que "moldean" su identidad. Se menciona la creación de un "perfil digital" que a menudo difiere del yo real.
* **Fluidez:** Se encontraron múltiples menciones a la identidad como algo cambiante y adaptable ("mi vida es un río en constante cambio"), reflejando la falta de solidez y compromiso a largo plazo descrita por Bauman.
* **Autenticidad vs. Performance:** El sistema diferenció claramente entre discursos auténticos (expresión de sentimientos genuinos, vulnerabilidad) y discursos performativos (creación de imagen pública, manipulación del tono para ganar *engagement*).

### 3.3. Eje: Cultura del Rendimiento y Burnout (Byung-Chul Han)
La evidencia empírica respalda fuertemente la tesis de la **Sociedad del Cansancio**:
* **Emociones Predominantes:** Las emociones más frecuentes asociadas al entorno digital son ansiedad, frustración, dolor físico (fatiga) y un sentimiento de "alojamiento" (no pertenencia).
* **Autoexplotación:** Se detectaron patrones de conducta donde el usuario se culpa a sí mismo por no ser productivo o visible. El uso de lenguaje productivo ("estoy haciendo todo lo posible") se mezcla con la falta de descanso, validando la idea de que el sujeto se explota a sí mismo creyendo ser libre.

### 3.4. Eje: Autonomía y Control Tecnológico (Foucault, Heidegger, Habermas)
* **Percepción de Autonomía:** La mayoría de los usuarios percibe su autonomía como **condicionada**. Frases como "no sé si la tecnología nos libera o nos convierte en espectadores" son comunes.
* **Vigilancia (Panóptico):** Bajo la óptica de Foucault, el sistema interpretó la sensación de los usuarios de ser "observados" y "juzgados" constantemente como una internalización del control (panóptico digital), donde modifican su conducta para agradar al algoritmo.
* **Desocultamiento (Heidegger):** La tecnología transforma la vida en un recurso ("datos"). Los usuarios expresan sentirse como en un "experimento" o un "juego de video", lo que indica una desrealización y una conversión del ser humano en *stock* de información.
* **Espacio Público (Habermas):** Se evidencia un debilitamiento del debate racional. Predomina la comunicación personal y emocional sobre la conciencia colectiva política, confirmando la fragmentación de la esfera pública.

---

## 4. Visualizaciones Semánticas

A partir de la minería de datos realizada por el sistema RAG, se estructuran los siguientes datos para su representación visual:

### 4.1. Nube de Palabras (Crisis de Sentido)
*Términos más recurrentes detectados en el corpus:*
1. Ansiedad
2. Vacío
3. Presión
4. Máscara / Filtro
5. Algoritmo
6. Soledad
7. Cansancio
8. Inseguridad
9. Conexión
10. Rendimiento

### 4.2. Gráfico de Distribución de Sentimientos
*Análisis emocional sobre el "Futuro" y "Propósito":*
* **Negativo/Ansioso (~60%):** Miedo, incertidumbre económica, preocupación climática, sensación de no ser suficiente.
* **Neutro/Reflexivo (~25%):** Cuestionamiento filosófico, duda metódica sobre la tecnología.
* **Positivo/Esperanza (~15%):** Conexión comunitaria, activismo social, deseo de cambio.

### 4.3. Mapa Conceptual de Conexiones
El análisis de co-ocurrencia de términos sugiere las siguientes relaciones:
* **Burnout** → se conecta con **Autoexplotación** y **Comparación Social**.
* **Algoritmo** → se conecta con **Identidad** y **Pérdida de Control**.
* **Vacío** → se conecta con **Scroll Infinito** y **Falta de Propósito**.

---

## 5. Comparativa Generacional (Millennials vs Gen Z)

El análisis del sistema RAG arrojó diferencias significativas en la percepción de valores:

| Aspecto | Generaciones Anteriores (según percepción Gen Z) | Generación Z (Hallazgos del Dataset) |
| :--- | :--- | :--- |
| **Éxito** | Estabilidad, bienes materiales, carrera lineal. | Autenticidad, salud mental, flexibilidad. |
| **Identidad** | Sólida, definida por el trabajo. | Líquida, definida por la expresión y el cambio. |
| **Tecnología** | Herramienta de uso. | Entorno vital (extensión del yo). |
| **Rechazo** | La Gen Z rechaza la "perfección" y la "productividad tóxica" que asocian a sus padres o jefes. | Valoran la vulnerabilidad y la experiencia vivencial sobre el estatus. |

---

## 6. Conclusiones Filosóficas Finales

**Confirmación de la Hipótesis:**
Existe evidencia empírica suficiente en el dataset para afirmar que la Generación Z atraviesa una **crisis de sentido estructural**. Los datos muestran una alta frecuencia de miedo, frustración y desesperanza, traducidos en una sensación de vacío existencial que no encuentra respuesta en las dinámicas digitales actuales.

**Integración Teórica:**
* **La Liquidez como Norma:** La crisis se agrava por la **Identidad Líquida (Bauman)**. Al no haber estructuras sólidas, la ansiedad se vuelve crónica ante la necesidad de redefinirse constantemente para el algoritmo.
* **El Cansancio del Yo:** La **Sociedad del Cansancio (Han)** se manifiesta plenamente. El sujeto ya no es oprimido por un jefe externo, sino por su propio "Yo ideal" digital, generando burnout y depresión por autoexplotación.
* **Autonomía y Mala Fe:** Se observa un fenómeno de **Mala Fe (Sartre)**: los usuarios son conscientes de la manipulación algorítmica, pero a menudo renuncian a su libertad ("el algoritmo me hizo ver esto") para evitar la angustia de la elección, convirtiéndose en sujetos pasivos o "datos" dentro del sistema.

**Veredicto sobre la Autonomía:**
No estamos ante una generación plenamente libre. Si bien hay una búsqueda discursiva de libertad, la praxis digital revela sujetos **autoexplotados**. La autonomía está condicionada tecnológicamente; la libertad se ejerce dentro de los límites que el algoritmo permite ver (burbuja de filtros), lo que Heidegger llamaría una reducción del ser humano a recurso disponible.

---

## 7. Anexos

### 7.1. Glosario Filosófico-Digital (Generado por IA)

* **Vacío Existencial:** Sensación de falta de propósito causada por la saturación de opciones superficiales y la falta de guías claras (metarrelatos). Usado por la Gen Z como "sentirse perdido".
* **Identidad Líquida:** Naturaleza cambiante e inestable del "yo" en redes. Capacidad de ser alguien distinto en cada plataforma, generando inseguridad.
* **Autoexplotación:** Coacción interna para rendir, producir contenido y estar conectado 24/7. El usuario es su propio tirano.
* **Panóptico Digital:** Estado de vigilancia permanente por parte de la comunidad y el algoritmo, que normaliza conductas y censura la autenticidad.
* **Mala Fe Algorítmica:** Autoengaño mediante el cual el usuario atribuye sus decisiones de consumo al algoritmo para no asumir la responsabilidad de su propio tiempo y atención.

### 7.2. Evidencia del Sistema
![Evidencia Sistema RAG](Proyecto3\evidencias\evidencia_sistena.jpeg)