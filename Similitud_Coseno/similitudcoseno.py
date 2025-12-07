import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset_sintetico_5000_ampliado.csv', encoding='latin-1', on_bad_lines='skip')

text_cols = [c for c in df.select_dtypes(include='object').columns]
col_scores = {c: df[c].fillna('').astype(str).map(len).mean() for c in text_cols}
text_col = max(col_scores, key=col_scores.get)
df[text_col] = df[text_col].fillna('').astype(str)

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df[text_col])

sim_matrix = cosine_similarity(X)

def top_k_similar(doc_index, k=5):
    sims = sim_matrix[doc_index].copy()
    sims[doc_index] = -1
    idx = np.argsort(sims)[::-1][:k]
    return df.iloc[idx][text_col], sims[idx]

def query_sim(query, k=5):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    idx = np.argsort(sims)[::-1][:k]
    return df.iloc[idx][text_col], sims[idx]

print("Columna analizada:", text_col)
print("\nDocumentos más similares al documento 0:\n")

texts, scores = top_k_similar(0, 5)
for t, s in zip(texts, scores):
    print("Score:", s)
    print(t)
    print("-----")

print("\nResultados de búsqueda similitud para: 'ejemplo de búsqueda'\n")

texts, scores = query_sim("ejemplo de búsqueda", 5)
for t, s in zip(texts, scores):
    print("Score:", s)
    print(t)
    print("-----")

plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix, cmap='viridis')
plt.title("Matriz de Similitud Coseno (CountVectorizer + Stopwords)")
plt.xlabel("Documentos")
plt.ylabel("Documentos")
plt.show()