# %% [markdown]
# # Laboratorio: Procesamiento de Lenguaje Natural y Minería de Texto
# Requisitos previos en terminal:
# pip install nltk spacy scikit-learn transformers torch sentence-transformers
# python -m spacy download es_core_news_sm

# %%
import re
import math
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Descarga de recursos básicos de NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Cargar modelo de spaCy para español
nlp = spacy.load("es_core_news_sm")

corpus_ejemplo = [
    "La inteligencia artificial está transformando el análisis de datos masivos en tiempo real.",
    "El procesamiento del lenguaje natural permite a las máquinas comprender el texto humano.",
    "Los modelos de lenguaje avanzados generan resúmenes y traducen textos con alta precisión."
]
texto_base = corpus_ejemplo[0]

# %% 1. Normalización de texto
def normalizar(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

texto_normalizado = normalizar(texto_base)
print("1. Normalización:\n", texto_normalizado)

# %% 2. Stemming
stemmer = SnowballStemmer('spanish')
palabras = texto_normalizado.split()
stems = [stemmer.stem(p) for p in palabras]
print("\n2. Stemming:\n", stems)

# %% 3. Eliminación de Stopwords
stop_words_es = set(stopwords.words('spanish'))
palabras_sin_sw = [p for p in palabras if p not in stop_words_es]
print("\n3. Sin Stopwords:\n", palabras_sin_sw)

# %% 4. Term Frequency (TF)
tf_conteo = Counter(palabras)
total_palabras = len(palabras)
tf_scores = {palabra: conteo / total_palabras for palabra, conteo in tf_conteo.items()}
print("\n4. Term Frequency (primeras 4 palabras):\n", dict(list(tf_scores.items())[:4]))

# %% 5. Inverse Document Frequency (IDF) y TF-IDF
vectorizador = TfidfVectorizer()
matriz_tfidf = vectorizador.fit_transform(corpus_ejemplo)
vocabulario = vectorizador.get_feature_names_out()
idfs = dict(zip(vocabulario, vectorizador.idf_))
print("\n5. IDF (muestra):\n", dict(list(idfs.items())[:4]))

# %% 6. Part-of-Speech (POS Tagging), 7. Lematización, 8. Parsing y 9. Named-entity (NER)
doc = nlp("El presidente de Colombia anunció nuevas inversiones en tecnología en Bogotá.")

print("\n6. POS Tagging:")
for token in doc[:6]:
    print(f"   {token.text} -> {token.pos_} ({token.tag_})")

print("\n7. Lematización:")
for token in doc[:6]:
    print(f"   {token.text} -> Lema: {token.lemma_}")

print("\n8. Dependency Parsing:")
for token in doc[:6]:
    print(f"   {token.text} --({token.dep_})--> {token.head.text}")

print("\n9. Named Entity Recognition (NER):")
for ent in doc.ents:
    print(f"   Entidad: '{ent.text}' | Etiqueta: {ent.label_}")

# %% 10. Generación de texto con LLM / 15. Text Generation
generador_llm = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
prompt_llm = "El futuro de la inteligencia artificial será"
salida_gen = generador_llm(prompt_llm, max_new_tokens=40, do_sample=True, temperature=0.7)
print("\n10 & 15. Generación de texto (LLM):\n", salida_gen[0]['generated_text'])

# %% 11. Question Answering
qa_pipeline = pipeline("question-answering", model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es")
contexto_qa = "El Procesamiento del Lenguaje Natural es una rama de la inteligencia artificial nacida en los años 1950."
pregunta_qa = "¿Cuándo nació el Procesamiento del Lenguaje Natural?"
res_qa = qa_pipeline(question=pregunta_qa, context=contexto_qa)
print(f"\n11. Question Answering:\n   Pregunta: {pregunta_qa}\n   Respuesta: {res_qa['answer']}")

# %% 12. Summarization
resumidor = pipeline("summarization", model="mrm8488/bert2bert_shared-spanish-finetuned-summarization")
texto_largo = (
    "Las redes neuronales artificiales son un modelo computacional inspirado en el funcionamiento del cerebro humano. "
    "Consisten en un conjunto de nodos interconectados organizados en capas. A través del entrenamiento sobre grandes "
    "volúmenes de datos, estas redes pueden detectar patrones complejos y realizar predicciones muy precisas."
)
resumen = resumidor(texto_largo, max_length=30, min_length=10, do_sample=False)
print("\n12. Resumen:\n", resumen[0]['summary_text'])

# %% 13. Sentence Similarity
modelo_sim = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
emb1 = modelo_sim.encode("El coche avanza veloz por la autopista.", convert_to_tensor=True)
emb2 = modelo_sim.encode("El auto va rápido en la carretera.", convert_to_tensor=True)
similitud = util.cos_sim(emb1, emb2).item()
print(f"\n13. Similitud de oraciones (Coseno): {similitud:.4f}")

# %% 14. Text Classification
x_train = [
    "Excelente producto, llegó a tiempo y de gran calidad",
    "Pésimo servicio, el artículo llegó roto y tarde",
    "Me encantó la atención y la compra fue rápida",
    "Muy mala experiencia, no lo recomiendo para nada"
]
y_train = ["Positivo", "Negativo", "Positivo", "Negativo"]

vec_clf = TfidfVectorizer()
X_vec = vec_clf.fit_transform(x_train)
clf = LogisticRegression().fit(X_vec, y_train)

texto_prueba = ["El servicio al cliente fue horrible"]
prediccion = clf.predict(vec_clf.transform(texto_prueba))
print(f"\n14. Clasificación de texto:\n   Entrada: '{texto_prueba[0]}' -> Predicción: {prediccion[0]}")

# %% 16. Translation
traductor = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
traduccion = traductor("Este laboratorio de procesamiento de texto está completo.")
print("\n16. Traducción (ES -> EN):\n", traduccion[0]['translation_text'])

# %% 17. Text Mining (Extracción de n-gramas más frecuentes)
vectorizador_ngrams = TfidfVectorizer(ngram_range=(2, 2))
matriz_ngrams = vectorizador_ngrams.fit_transform(corpus_ejemplo)
bigramas = vectorizador_ngrams.get_feature_names_out()
print("\n17. Text Mining (Extracción de Bigramas representativos):\n", list(bigramas)[:5])