import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# 1. Cargar el dataset que tenías con errores
# Asegúrate de que el nombre del archivo coincida con el que tienes
df = pd.read_csv('dataset_aumentado_final.csv')

# 2. Eliminar filas vacías o corruptas
# Si falta el 'id' o el 'texto', consideramos la fila inservible
df_clean = df.dropna(subset=['id', 'texto']).copy()

# 3. Rellenar fechas faltantes (NaN)
# Primero convertimos la columna a formato fecha real para poder trabajar con ella
df_clean['fecha'] = pd.to_datetime(df_clean['fecha'], errors='coerce')

# Obtenemos el rango de fechas válido (mínimo y máximo) de los datos que sí tienen fecha
valid_dates = df_clean['fecha'].dropna()
if not valid_dates.empty:
    start_date = valid_dates.min()
    days_range = (valid_dates.max() - start_date).days
else:
    # Si no hubiera ninguna fecha válida, usamos un rango por defecto (2022-2024)
    start_date = datetime(2022, 1, 1)
    days_range = 700

# Definimos una pequeña función para generar fechas aleatorias
def random_date(start, range_days):
    return start + timedelta(days=random.randint(0, range_days))

# Identificamos qué filas no tienen fecha
mask_missing_dates = df_clean['fecha'].isna()

# A esas filas les asignamos una fecha aleatoria dentro del rango calculado
df_clean.loc[mask_missing_dates, 'fecha'] = df_clean.loc[mask_missing_dates].apply(
    lambda row: random_date(start_date, days_range), axis=1
)

# Volvemos a dar formato de texto YYYY-MM-DD para que se guarde bien en CSV
df_clean['fecha'] = df_clean['fecha'].dt.strftime('%Y-%m-%d')

# 4. Guardar el resultado final limpio
df_clean.to_csv('dataset_final_corregido.csv', index=False)

print(f"¡Listo! Archivo guardado con {len(df_clean)} filas limpias.")