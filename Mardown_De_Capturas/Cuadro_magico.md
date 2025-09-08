# Práctica: Verificación de un Cuadrado Mágico

## Objetivo
Comprobar que una matriz dada es un cuadrado mágico, es decir, que la suma de los elementos de cada fila, cada columna y cada diagonal principal es la misma (constante mágica).


### Constante Mágica
La constante mágica se calcula con la fórmula:

M=n(n2+1)​/2


### Paso 1: Identificar el Cuadrado Mágico
El cuadrado mágico es la submatriz interior de  3 x 3 :
8 1 6
3 5 7
4 9 2


### Paso 2: Calcular la Constante Mágica
Para  n = 3 :

M=n(n2+1)​/2



### Paso 3: Verificar Filas
- Fila 1:  8 + 1 + 6 = 15 
- Fila 2:  3 + 5 + 7 = 15 
- Fila 3:  4 + 9 + 2 = 15 

### Paso 4: Verificar Columnas
- Columna 1:  8 + 3 + 4 = 15 
- Columna 2:  1 + 5 + 9 = 15 
- Columna 3:  6 + 7 + 2 = 15 

### Paso 5: Verificar Diagonales
- Diagonal principal:  8 + 5 + 2 = 15 
- Diagonal secundaria:  6 + 5 + 4 = 15 

### Conclusión
Todas las sumas son iguales a 15, por lo que el cuadrado interior de \( 3 \times 3 \).



