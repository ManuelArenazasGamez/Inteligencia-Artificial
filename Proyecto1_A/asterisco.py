import pygame
import math
from queue import PriorityQueue


ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización de Algoritmo A*")

# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)      # Nodos "abiertos" (en el open_set)
ROJO = (255, 0, 0)        # Nodos "cerrados" (ya visitados)
NARANJA = (255, 165, 0)   # Nodo de inicio
PURPURA = (128, 0, 128)   # Nodo final
TURQUESA = (64, 224, 208) # Nodos del camino final

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.vecinos = []

    def get_pos(self):
        return self.fila, self.col

    # --- Funciones de estado (boolean) ---
    def es_cerrado(self):
        return self.color == ROJO

    def es_abierto(self):
        return self.color == VERDE

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    # --- Funciones de acción (cambiar color) ---
    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_cerrado(self):
        self.color = ROJO

    def hacer_abierto(self):
        self.color = VERDE

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA
    
    def hacer_camino(self):
        self.color = TURQUESA

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

    def actualizar_vecinos(self, grid):
        """ Llena la lista self.vecinos con nodos transitables """
        self.vecinos = []
        # ABAJO
        if self.col < self.total_filas - 1 and not grid[self.fila][self.col + 1].es_pared():
            self.vecinos.append(grid[self.fila][self.col + 1])
        # ARRIBA
        if self.col > 0 and not grid[self.fila][self.col - 1].es_pared():
            self.vecinos.append(grid[self.fila][self.col - 1])
        # DERECHA
        if self.fila < self.total_filas - 1 and not grid[self.fila + 1][self.col].es_pared():
            self.vecinos.append(grid[self.fila + 1][self.col])
        # IZQUIERDA
        if self.fila > 0 and not grid[self.fila - 1][self.col].es_pared():
            self.vecinos.append(grid[self.fila - 1][self.col])

    def __lt__(self, other):
        """ Permite comparar dos nodos (necesario para PriorityQueue) """
        return False

# --- Funciones del Algoritmo A* ---

def h(p1, p2):
    """
    Calcula la heurística (distancia Manhattan) entre dos puntos.
    p1 y p2 son tuplas (fila, col)
    """
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruir_camino(came_from, actual, dibujar):
    """ Dibuja el camino final retrocediendo desde el nodo final """
    while actual in came_from:
        actual = came_from[actual]
        actual.hacer_camino()
        dibujar()

def algoritmo_a_estrella(dibujar, grid, inicio, fin):
    """
    La lógica principal del algoritmo A*.
    'dibujar' es una función lambda para actualizar la pantalla.
    """
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, inicio)) # (f_score, count, nodo)
    
    came_from = {} # Diccionario para reconstruir el camino
    
    # g_score: Costo real desde el inicio hasta este nodo
    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    g_score[inicio] = 0
    
    # f_score: Costo estimado (g_score + heurística)
    f_score = {nodo: float("inf") for fila in grid for nodo in fila}
    f_score[inicio] = h(inicio.get_pos(), fin.get_pos())

    # Para saber qué hay en el open_set (PriorityQueue no lo permite fácilmente)
    open_set_hash = {inicio} 

    while not open_set.empty():
        # Permitir al usuario cerrar la ventana mientras corre el algoritmo
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Obtener el nodo con el f_score más bajo
        actual = open_set.get()[2] 
        open_set_hash.remove(actual)

        # --- ¡Objetivo encontrado! ---
        if actual == fin:
            reconstruir_camino(came_from, fin, dibujar)
            fin.hacer_fin() # Asegurarse de que el fin siga púrpura
            inicio.hacer_inicio() # Y el inicio naranja
            return True

        # --- Procesar vecinos ---
        for vecino in actual.vecinos:
            # Asumimos que el costo para moverse a un vecino es 1
            temp_g_score = g_score[actual] + 1 

            if temp_g_score < g_score[vecino]:
                # Se encontró un camino mejor hacia este vecino
                came_from[vecino] = actual
                g_score[vecino] = temp_g_score
                f_score[vecino] = temp_g_score + h(vecino.get_pos(), fin.get_pos())
                
                if vecino not in open_set_hash:
                    count += 1
                    open_set.put((f_score[vecino], count, vecino))
                    open_set_hash.add(vecino)
                    vecino.hacer_abierto() # Pintar de verde (abierto)

        dibujar() # Actualizar la pantalla en cada paso

        # Marcar el nodo actual como visitado (cerrado)
        if actual != inicio:
            actual.hacer_cerrado() # Pintar de rojo (cerrado)

    return False # No se encontró un camino

# --- Funciones de Dibujo (sin cambios) ---

def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)
    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    y, x = pos # OJO: En pygame pos es (x, y)
    
    # Tu código original tenía (y, x = pos) que significa y=pos[0], x=pos[1]
    # Lo corrijo a la forma estándar de pygame:
    x_pixel, y_pixel = pos
    
    # Tu lógica de grid usa (fila, col) como (x, y)
    fila = x_pixel // ancho_nodo
    col = y_pixel // ancho_nodo
    
    # Hago una comprobación para evitar IndexError si se hace clic fuera
    if 0 <= fila < filas and 0 <= col < filas:
        return fila, col
    return None # Retornar None si el clic está fuera de los límites

def main(ventana, ancho):
    FILAS = 11
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None

    corriendo = True
    algoritmo_iniciado = False

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            # No permitir clicks si el algoritmo ya corrió
            if algoritmo_iniciado:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    # Resetear todo con 'c'
                    grid = crear_grid(FILAS, ancho)
                    inicio = None
                    fin = None
                    algoritmo_iniciado = False
                continue # Ignorar otros eventos post-algoritmo

            # --- Manejo de Clicks ---
            if pygame.mouse.get_pressed()[0]:  # Click izquierdo
                pos = pygame.mouse.get_pos()
                click_pos = obtener_click_pos(pos, FILAS, ancho)
                if not click_pos: continue # Ignorar clics fuera del grid
                
                fila, col = click_pos
                nodo = grid[fila][col]
                
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()
                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()
                elif nodo != fin and nodo != inicio:
                    nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:  # Click derecho
                pos = pygame.mouse.get_pos()
                click_pos = obtener_click_pos(pos, FILAS, ancho)
                if not click_pos: continue

                fila, col = click_pos
                nodo = grid[fila][col]
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

            # --- Manejo de Teclado ---
            if event.type == pygame.KEYDOWN:
                # Iniciar el algoritmo con ESPACIO
                if event.key == pygame.K_SPACE and inicio and fin:
                    algoritmo_iniciado = True
                    # Actualizar los vecinos de todos los nodos ANTES de correr
                    for fila_grid in grid:
                        for nodo in fila_grid:
                            nodo.actualizar_vecinos(grid)
                    
                    # Llamar al algoritmo
                    algoritmo_a_estrella(
                        lambda: dibujar(ventana, grid, FILAS, ancho), # Función de dibujo
                        grid, 
                        inicio, 
                        fin
                    )

                # Resetear el grid con 'C'
                if event.key == pygame.K_c:
                    grid = crear_grid(FILAS, ancho)
                    inicio = None
                    fin = None
                    algoritmo_iniciado = False


    pygame.quit()

if __name__ == "__main__":
    main(VENTANA, ANCHO_VENTANA)