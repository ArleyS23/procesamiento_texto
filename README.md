# Laboratorio de Procesamiento de Lenguaje Natural

Notebook académico en Python para estudiar técnicas fundamentales de Procesamiento de Lenguaje Natural (NLP) con ejemplos en español. El proyecto está preparado para ejecutarse en Google Colab o Jupyter.

## Objetivos

- Normalizar y transformar textos en español.
- Analizar frecuencia y relevancia de términos.
- Aplicar análisis lingüístico con spaCy.
- Utilizar modelos preentrenados de Hugging Face.
- Comparar métodos estadísticos, lingüísticos y neuronales.
- Interpretar los resultados y sus limitaciones.

## Técnicas incluidas

El notebook cubre, en orden:

1. Normalización de texto con minúsculas, limpieza y expresiones regulares.
2. Stemming con `SnowballStemmer` para español.
3. Eliminación de stopwords con NLTK y respaldo local.
4. Term Frequency (TF).
5. Inverse Document Frequency (IDF) con `TfidfVectorizer`.
6. Part-of-speech tagging con spaCy.
7. Lematización con spaCy.
8. Dependency Parsing.
9. Named-Entity Recognition (NER).
10. Generación de texto con una LLM.
11. Question Answering en español.
12. Summarization.
13. Sentence Similarity con embeddings y similitud coseno.
14. Text Classification con TF-IDF y Logistic Regression.
15. Traducción español-inglés.
16. Generación de texto con bigramas.
17. Text mining mediante extracción de n-gramas.

Además, se incluye un corpus tabular con comentarios identificados, tablas de transformaciones, análisis lingüístico por comentario, términos frecuentes, valores IDF, bigramas y un resumen integrado de resultados.

## Archivos

- `procesamiento_texto.ipynb`: notebook principal con explicaciones, código y resultados.
- `procesamiento_texto.py`: versión Python organizada por celdas con marcadores `# %%`.
- `README.md`: documentación del proyecto.

## Ejecución en Google Colab

1. Abre [Google Colab](https://colab.research.google.com/).
2. Selecciona **Archivo > Subir notebook**.
3. Sube `procesamiento_texto.ipynb`.
4. Ejecuta la primera celda de instalación.
5. Si Colab solicita reiniciar la sesión, acepta y ejecuta nuevamente la celda de configuración.
6. Ejecuta las celdas restantes en orden.

La primera ejecución requiere conexión a internet para instalar paquetes y descargar modelos. El notebook detecta automáticamente si hay GPU disponible y utiliza CPU como alternativa.

## Instalación local

```bash
pip install -U nltk pandas spacy scikit-learn transformers sentence-transformers
python -m spacy download es_core_news_sm
```

También se puede abrir el archivo `.ipynb` directamente en Jupyter Notebook, JupyterLab o VS Code con la extensión de Jupyter.

## Modelos utilizados

- spaCy: `es_core_news_sm`
- Hugging Face: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Question Answering: `mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es`
- Summarization: `mrm8488/bert2bert_shared-spanish-finetuned-summarization`
- Sentence Similarity: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Translation: `Helsinki-NLP/opus-mt-es-en`

## Limitaciones

El corpus y el conjunto de entrenamiento son pequeños y tienen finalidad académica. Los resultados de clasificación no representan una evaluación de producción. Las respuestas generativas, traducciones y resúmenes deben revisarse porque pueden contener errores u omisiones. El tiempo de ejecución depende de la conexión, la memoria disponible y el uso de CPU o GPU.

## Licencia académica

Material elaborado con fines educativos y de demostración de técnicas de NLP en Python.
