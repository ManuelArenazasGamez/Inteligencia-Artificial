import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# --- PARTE 1: LEER TUS DATOS MANUALES ---

archivo_de_tweets = "tweets.txt"
lista_de_tweets_reales = []

print(f"Leyendo tweets desde el archivo: {archivo_de_tweets}")

try:
    with open(archivo_de_tweets, 'r', encoding='utf-8') as f:
        for linea in f:
            # strip() quita los espacios en blanco y saltos de línea
            if linea.strip(): 
                lista_de_tweets_reales.append(linea.strip())
    
    print(f"Se cargaron {len(lista_de_tweets_reales)} tuits desde tu archivo.")

except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo '{archivo_de_tweets}'.")
    print("Asegúrate de haber creado y guardado el archivo en la misma carpeta.")
    exit()
except Exception as e:
    print(f"Ocurrió un error al leer el archivo: {e}")
    exit()

# --- PARTE 2: EL MISMO ANÁLISIS DE TEMAS (LDA) DE ANTES ---

if lista_de_tweets_reales:
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Descargando la lista de 'stopwords' de NLTK...")
        nltk.download('stopwords')

    # 1. Pre-procesamiento y Vectorización
    stop_words_es = nltk.corpus.stopwords.words('spanish')
    
    stop_words_es.extend([
        'q', 'k', 'https', 'co', 'RT', 'rt', 'vía', 'ser', 'van', 'va', 'así', 'solo',
        'tec', 'michoacán', 'morelia', 'tecnológico', 'itmorelia',
        'estudiantes', 'ingeniería', 'gobierno'
    ])

    vectorizer = CountVectorizer(max_df=0.8, min_df=2, stop_words=stop_words_es)
    
    try:
        X = vectorizer.fit_transform(lista_de_tweets_reales)
    except ValueError:
        print("\nANÁLISIS DETENIDO: No hay suficientes palabras (vocabulario) para analizar.")
        print("Intenta recolectar más tuits (idealmente +50) o palabras únicas.")
        exit()

    # 2. Aplicar el modelo LDA
    n_temas = 3
    lda = LatentDirichletAllocation(n_components=n_temas, random_state=42)
    lda.fit(X)

    # 3. Mostrar los resultados
    print("\n--- TEMAS ENCONTRADOS (basado en tu archivo manual) ---")
    feature_names = vectorizer.get_feature_names_out()

    for topico_idx, topico in enumerate(lda.components_):
        print(f"\nTema #{topico_idx + 1}:")
        top_features = [feature_names[i] for i in topico.argsort()[:-6:-1]]
        print(" ".join(top_features))