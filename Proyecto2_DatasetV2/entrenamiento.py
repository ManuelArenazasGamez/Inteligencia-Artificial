import numpy as np
import os
import re
import cv2  # OpenCV para procesar imágenes
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.layers import RandomContrast
# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

DIRNAME = r'C:\Users\Admin\Documents\Repos\Inteligencia-Artificial\Proyecto2_DatasetV2\Animals'

IMG_WIDTH = 100
IMG_HEIGHT = 100
IMG_CHANNELS = 3


EPOCHS = 80
BATCH_SIZE = 32

# ==========================================
# 2. CARGA Y PREPROCESAMIENTO DE IMÁGENES
# ==========================================
images = []
labels = []
class_names = []
class_indices = {}

print(f"Iniciando lectura de imágenes en: {DIRNAME}")

# Obtenemos las clases basadas en las carpetas (ant, cats, dogs, etc.)
carpetas = sorted([d for d in os.listdir(DIRNAME) if os.path.isdir(os.path.join(DIRNAME, d))])
print(f"Clases detectadas: {carpetas}")

for idx, folder in enumerate(carpetas):
    class_names.append(folder)
    class_indices[folder] = idx
    
    folder_path = os.path.join(DIRNAME, folder)
    
    count = 0
    for filename in os.listdir(folder_path):
        # Filtramos solo imágenes
        if re.search(r"\.(jpg|jpeg|png|bmp|tiff|webp)$", filename, re.IGNORECASE):
            filepath = os.path.join(folder_path, filename)
            
            # Leer y redimensionar
            image = cv2.imread(filepath)
            
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                
                images.append(image)
                labels.append(idx) 
                count += 1
    
    print(f" - Carpeta '{folder}': {count} imágenes cargadas.")

print(f"\nTotal de imágenes cargadas: {len(images)}")

# ==========================================
# 3. PREPARACIÓN DE DATOS (ARRAYS)
# ==========================================
X = np.array(images, dtype=np.uint8)
y = np.array(labels)

# Limpiamos memoria
del images

# Normalización (0 a 1) y conversión a float
X = X.astype('float32') / 255.0

# One-hot encoding 
y_one_hot = to_categorical(y)

# Separar en Entrenamiento (80%) y Prueba (20%)
train_X, test_X, train_Y, test_Y = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

print('Datos de entrenamiento:', train_X.shape)
print('Datos de prueba:', test_X.shape)

# ==========================================
# 4. DEFINICIÓN DEL MODELO (CNN)
# ==========================================
nClasses = len(class_names)

# Definimos las transformaciones aleatorias
data_augmentation = Sequential([
    RandomFlip("horizontal"),   
    RandomRotation(0.1),        
    RandomZoom(0.1),
    RandomContrast(0.2),             
])

animal_model = Sequential()

animal_model.add(data_augmentation)
# Capa 1
animal_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)))
animal_model.add(LeakyReLU(alpha=0.1))
animal_model.add(MaxPooling2D((2, 2), padding='same'))
animal_model.add(Dropout(0.25))

# Capa 2
animal_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
animal_model.add(LeakyReLU(alpha=0.1))
animal_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
animal_model.add(Dropout(0.25))

# Capa 3
animal_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
animal_model.add(LeakyReLU(alpha=0.1))
animal_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
animal_model.add(Dropout(0.4))

# Flatten y Dense
animal_model.add(Flatten())
animal_model.add(Dense(128, activation='linear'))
animal_model.add(LeakyReLU(alpha=0.1))
animal_model.add(Dropout(0.3))

# Salida
animal_model.add(Dense(nClasses, activation='softmax'))

animal_model.summary()

# Compilación
animal_model.compile(loss=keras.losses.categorical_crossentropy, 
                     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     metrics=['accuracy'])

# ==========================================
# 5. ENTRENAMIENTO (CON EARLY STOPPING)
# ==========================================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

print("\n>>> Iniciando entrenamiento...")
history = animal_model.fit(
    train_X, train_Y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(test_X, test_Y),
    callbacks=[early_stop]
)

# ==========================================
# 6. EVALUACIÓN Y GUARDADO
# ==========================================
print("\n>>> Evaluando modelo...")
test_eval = animal_model.evaluate(test_X, test_Y, verbose=0)
print(f'Pérdida (Loss) en test: {test_eval[0]:.4f}')
print(f'Precisión (Accuracy) en test: {test_eval[1]:.4f}')

# Reporte detallado
predicted_classes = animal_model.predict(test_X)
predicted_ids = np.argmax(predicted_classes, axis=1)
true_ids = np.argmax(test_Y, axis=1)

print("\nReporte de Clasificación:")
print(classification_report(true_ids, predicted_ids, target_names=class_names))

# Guardar el modelo
save_path = 'modelo_animalesocho.keras'
animal_model.save(save_path)
print(f"\n>>> ¡Entrenamiento finalizado! Modelo guardado como '{save_path}'")