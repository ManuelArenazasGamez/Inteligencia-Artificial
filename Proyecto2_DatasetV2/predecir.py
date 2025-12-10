import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from tensorflow.keras.models import load_model

# 1. CARGA EL MODELO
modelo_h5 = 'Modelos/modelo_animalesocho.keras'
try:
    animal_model = load_model(modelo_h5)
    print("Modelo cargado exitosamente.")
except:
    print(f"Error: No se encontró '{modelo_h5}'.")
    exit()

carpeta_pruebas = r'C:\Users\manue\OneDrive\Documentos\9° Semestre\Inteligencia-Artificial\Proyecto2_DatasetV2\Pruebas'

# Etiquetas
animales = ['ant', 'cats', 'dogs', 'ladybug', 'turtles']

print(f"\nProcesando imágenes de: {carpeta_pruebas}...\n")

# 3. LEER, PREDECIR Y MOSTRAR
archivos = os.listdir(carpeta_pruebas)

for archivo in archivos:
    if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
        ruta_completa = os.path.join(carpeta_pruebas, archivo)
        
        try:
            # Leer imagen original para mostrarla bonita
            image_original = plt.imread(ruta_completa)
            
            image_para_modelo = resize(image_original, (100, 100), anti_aliasing=True, clip=False, preserve_range=True)
            
            # Normalizar y dar formato
            X_test = np.array([image_para_modelo], dtype=np.uint8)
            X_test = X_test.astype('float32') / 255.0
            
            # Predecir
            prediccion = animal_model.predict(X_test, verbose=0)
            idx = np.argmax(prediccion)
            animal = animales[idx]
            
            plt.figure(figsize=(5, 5)) 
            plt.imshow(image_original) 
            plt.axis('off') 
            
            plt.title(f"Predicción: {animal.lower()}", fontsize=14, color='black')
            plt.show() 
            
            print(f"Archivo: {archivo} -> {animal}\n")
            
        except Exception as e:
            print(f"No se pudo leer {archivo}: {e}")

print("Fin de las pruebas.")