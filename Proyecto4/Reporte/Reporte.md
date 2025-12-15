# Reporte de Práctica: Fine-Tuning de un Tutor Inteligente de Algoritmos

**Institución:** Instituto Tecnológico de Morelia
**Carrera:** Ingeniería en Sistemas Computacionales
**Estudiante:** Manuel Arenazas Gámez
**Asignatura:** Inteligencia Artificial (Proyecto 4)
**Periodo:** Agosto - Diciembre 2025

-----

## 1\. Introducción

El aprendizaje de estructuras de datos y algoritmos representa uno de los mayores desafíos para los estudiantes de Ingeniería en Sistemas. La abstracción necesaria para comprender conceptos como la recursividad o la complejidad temporal a menudo requiere una atención personalizada que no siempre es escalable en un aula tradicional.

El presente proyecto documenta el desarrollo y entrenamiento de un **Tutor Inteligente de Algoritmos**. Utilizando técnicas avanzadas de **Fine-Tuning (Ajuste Fino)** sobre Modelos de Lenguaje Grande (LLMs), se ha creado un asistente virtual capaz de explicar conceptos, resolver ejercicios y depurar código con un enfoque pedagógico, adaptado a las necesidades específicas de los estudiantes del Tecnológico de Morelia.

## 2\. Planteamiento del Problema

Los estudiantes enfrentan dificultades significativas al estudiar algoritmos debido a:

  * Falta de explicaciones detalladas o contextualizadas con analogías.
  * Escasez de ejemplos paso a paso en su idioma nativo.
  * Dificultad para visualizar el funcionamiento interno (ej. Stack Overflow).
  * Ausencia de retroalimentación inmediata al cometer errores lógicos.

La solución propuesta es un sistema de tutoría basado en IA que no solo entregue código, sino que explique el *porqué* y el *cómo*, emulando la interacción con un profesor humano.

## 3\. Objetivos

### 3.1 Objetivo General

Entrenar y evaluar un modelo de lenguaje mediante **Fine-Tuning supervisado** para que opere como un tutor especializado en enseñanza de algoritmos, capaz de brindar explicaciones comprensibles, detalladas y adaptadas a distintos niveles de dominio.

### 3.2 Objetivos Específicos

  * Diseñar y curar un dataset educativo en formato JSONL compuesto por explicaciones paso a paso y pares de instrucción-respuesta.
  * Implementar un pipeline de entrenamiento eficiente en memoria utilizando **QLoRA (Quantized Low-Rank Adaptation)** para permitir su ejecución en hardware local (Windows/GPU).
  * Ajustar hiperparámetros críticos (Learning Rate, Epochs, Temperature) para maximizar la adherencia al estilo pedagógico.
  * Evaluar el desempeño del modelo resultante en tareas de explicación de complejidad ($O(n)$) y depuración de código.

## 4\. Metodología

Para cumplir con los objetivos bajo restricciones de hardware local, se optó por una arquitectura de **Parameter-Efficient Fine-Tuning (PEFT)**.

### 4.1 Arquitectura del Modelo

  * **Modelo Base:** `microsoft/Phi-3-mini-4k-instruct`. Se seleccionó por su alta capacidad de razonamiento lógico y eficiencia (3.8 Billones de parámetros), ideal para tareas académicas.
  * **Técnica de Entrenamiento:** **QLoRA**. Se utilizó cuantización de 4-bits (`NF4`) para reducir la huella de memoria de la GPU, permitiendo el entrenamiento en una tarjeta gráfica de consumo.
  * **Adaptadores (LoRA):** Se inyectaron matrices de rango bajo ($r=16$, $\alpha=32$) apuntando a todos los módulos lineales (`target_modules="all-linear"`), lo que permite al modelo aprender patrones complejos sin modificar los pesos originales congelados.

### 4.2 Preparación del Dataset

Se creó un archivo `tutor_programacion.jsonl` conteniendo ejemplos curados con el siguiente formato **ChatML**, nativo para el modelo Phi-3:

```json
{"prompt": "¿Qué es la complejidad O(n)?", "response": "Imagina que tienes una baraja de cartas..."}
```

El preprocesamiento incluyó la tokenización y el formateo con etiquetas especiales: `<|user|>\n{pregunta}<|end|>\n<|assistant|>\n{respuesta}<|end|>`.

### 4.3 Configuración del Entrenamiento

El entrenamiento se ejecutó localmente utilizando las librerías `transformers`, `peft` y `bitsandbytes`. Se aplicó una estrategia de **Overfitting controlado** para forzar la adopción del estilo pedagógico:

  * **Épocas (Epochs):** 15 (Para asegurar la memorización del estilo tutor).
  * **Tasa de Aprendizaje (Learning Rate):** $2e-3$ (Agresiva para cambios significativos).
  * **Optimizador:** `adamw_8bit` (Para ahorro de memoria).
  * **Batch Size:** 1 (Con acumulación de gradiente de 4 pasos).

## 5\. Resultados

El proceso de Fine-Tuning se completó exitosamente, generando los adaptadores LoRA en la carpeta `./tutor_ajustado_phi3`.

### 5.1 Evidencia del Entrenamiento

La pérdida (Loss) del modelo mostró una tendencia descendente, indicando que la red neuronal estaba aprendiendo efectivamente a predecir las respuestas pedagógicas del dataset.


### 5.2 Evaluación Cualitativa (Chat con el Tutor)

Se realizaron pruebas de inferencia utilizando un script de Python con temperatura baja ($0.2$) para verificar la calidad de las respuestas.

#### Prueba 1: Conceptos Teóricos

**Pregunta:** "¿Por qué decimos que acceder a un elemento en un Array es O(1) pero en una Lista Enlazada es O(n)?"
**Respuesta del Modelo:**

> *"En un Array, la memoria está contiguamente asignada, por lo que sabes exactamente dónde está tu elemento. En una Lista Enlazada, debes recorrer desde la cabeza hasta llegar al nodo deseado."*

**Análisis:** El modelo proporcionó una definición técnica precisa y correcta, demostrando que retiene el conocimiento de estructuras de datos y lo explica con claridad.

#### Prueba 2: Depuración de Código

**Pregunta:** "Ayuda, mi código no se detiene nunca: `i = 0; while i < 5: print(i)`"
**Respuesta del Modelo:**

> *"Estás incrementando 'i' después de imprimir, pero no lo estás comparando correctamente. Deberías hacer 'i += 1' dentro del bucle."*

**Análisis:** El tutor identificó correctamente el error lógico (bucle infinito por falta de incremento) y ofreció la solución, cumpliendo el rol de asistente de programación.

![CHAT PROBAR.PY](Proyecto4\Reporte\evidencia_probar.png)

## 6\. Discusión

Durante la evaluación se observaron dos fenómenos interesantes:

1.  **Generalización del Estilo:** El modelo logró transferir el formato de respuesta ("Tutor: Pensando...") a preguntas de algoritmos que no estaban explícitamente en el dataset, demostrando capacidad de generalización en el dominio de la programación.
2.  **Limitaciones de Dominio:** Cuando se consultaron temas fuera del entrenamiento intensivo (como CSS o Historia), el modelo tendió a regresar a su comportamiento base, aunque intentó mantener la estructura del chat. Esto confirma que el Fine-Tuning con datasets pequeños (\<100 ejemplos) es efectivo para **transferencia de estilo** y **especialización**, pero no para la inyección masiva de conocimientos fácticos nuevos.

## 7\. Conclusión

El proyecto ha cumplido satisfactoriamente con el objetivo de desarrollar un **Tutor Inteligente de Algoritmos**. Se demostró que es técnicamente viable realizar un Fine-Tuning avanzado en un entorno local Windows utilizando técnicas de cuantización (QLoRA), superando las limitaciones de hardware tradicionales.

El modelo resultante, basado en Phi-3, es una herramienta funcional que puede asistir a estudiantes de Ingeniería en Sistemas, proveyendo explicaciones claras y ayudas de depuración las 24 horas del día. Como trabajo futuro, se propone ampliar el dataset a más de 1,000 ejemplos para cubrir más lenguajes de programación y fortalecer el uso de analogías específicas.

-----

Manuel Arenazas Gámez