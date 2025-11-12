import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException

# --- Configuración ---
# Asegúrate de tener 'chromedriver.exe' en la misma carpeta
# y que coincida con tu versión de Chrome
try:
    s = Service('./chromedriver.exe')
    driver = webdriver.Chrome(service=s)
except Exception as e:
    print(f"Error iniciando Selenium: {e}")
    print("Asegúrate de que 'chromedriver.exe' esté en la carpeta y sea la versión correcta.")
    exit()

# --- La Búsqueda ---
# Esta es la consulta de tu proyecto
query = '("Tecnológico de Morelia" OR "Tec de Morelia" OR #ITMorelia) lang:es'
url = f"https://x.com/search?q={query.replace(' ', '%20').replace('"', '%22')}&src=typed_query"
driver.get(url)

# --- Pausa para Iniciar Sesión Manualmente ---
print("--- TIENES 60 SEGUNDOS PARA INICIAR SESIÓN MANUALMENTE EN LA VENTANA DE CHROME ---")
print("Ve a la ventana que se acaba de abrir, inicia sesión y espera.")
time.sleep(180) # Aumenta si necesitas más tiempo

print("Iniciando recolección...")

# --- Hacer Scroll para Cargar Tweets ---
tweets_recolectados = set() # Usamos un 'set' para evitar duplicados
ids_tweets_vistos = set()
scroll_pauses = 5

for _ in range(scroll_pauses):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    print("Haciendo scroll...")
    time.sleep(3) # Esperar a que carguen

    # --- Extracción ---
    # Este selector es el corazón
    elementos_tweet = driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')
    
    for tweet in elementos_tweet:
        if tweet.id not in ids_tweets_vistos:
            ids_tweets_vistos.add(tweet.id)
            try:
                # Extrae el texto
                texto_tweet = tweet.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]').text
                if texto_tweet: # Asegurarse de que no esté vacío
                    tweets_recolectados.add(texto_tweet)
            except NoSuchElementException:
                pass # Ignorar anuncios, "quién seguir", etc.
            except Exception as e:
                print(f"Error menor extrayendo un tweet: {e}")

print(f"Se encontraron {len(tweets_recolectados)} tuits únicos después de {scroll_pauses} scrolls.")
driver.quit()

# --- Mostrar Resultados ---
print("\n--- Tweets Recolectados (Únicos) ---")
for i, texto in enumerate(tweets_recolectados):
    print(f"TWEET #{i+1}")
    print(texto)
    print("-" * 20)

