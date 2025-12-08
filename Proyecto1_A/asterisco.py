import pygame
import math
from queue import PriorityQueue


ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption(" Algoritmo A*")

# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)      
ROJO = (255, 0, 0)        
NARANJA = (255, 165, 0)   
PURPURA = (128, 0, 128)   
TURQUESA = (64, 224, 208) 

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
        self.vecinos = []
        
        # ABAJO
        if self.fila < self.total_filas - 1 and not grid[self.fila + 1][self.col].es_pared():
            self.vecinos.append(grid[self.fila + 1][self.col])
        # ARRIBA
        if self.fila > 0 and not grid[self.fila - 1][self.col].es_pared():
            self.vecinos.append(grid[self.fila - 1][self.col])
        # DERECHA
        if self.col < self.total_filas - 1 and not grid[self.fila][self.col + 1].es_pared():
            self.vecinos.append(grid[self.fila][self.col + 1])
        # IZQUIERDA
        if self.col > 0 and not grid[self.fila][self.col - 1].es_pared():
            self.vecinos.append(grid[self.fila][self.col - 1])

        # Abajo-Derecha
        if self.fila < self.total_filas - 1 and self.col < self.total_filas - 1 and not grid[self.fila + 1][self.col + 1].es_pared():
            self.vecinos.append(grid[self.fila + 1][self.col + 1])
        # Abajo-Izquierda
        if self.fila < self.total_filas - 1 and self.col > 0 and not grid[self.fila + 1][self.col - 1].es_pared():
            self.vecinos.append(grid[self.fila + 1][self.col - 1])
        # Arriba-Derecha
        if self.fila > 0 and self.col < self.total_filas - 1 and not grid[self.fila - 1][self.col + 1].es_pared():
            self.vecinos.append(grid[self.fila - 1][self.col + 1])
        # Arriba-Izquierda
        if self.fila > 0 and self.col > 0 and not grid[self.fila - 1][self.col - 1].es_pared():
            self.vecinos.append(grid[self.fila - 1][self.col - 1])

    def __lt__(self, other):
        """ Permite comparar dos nodos (necesario para PriorityQueue) """
        return False



def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    distancia = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distancia * 1.5

def reconstruir_camino(came_from, actual, dibujar):
    while actual in came_from:
        actual = came_from[actual]
        actual.hacer_camino()
        dibujar()

def algoritmo_a_estrella(dibujar, grid, inicio, fin):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, inicio)) # (f_score, count, nodo)
    
    came_from = {} # Diccionario para reconstruir el camino
    
    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    g_score[inicio] = 0
    
    # f_score: Costo estimado (g_score + heurística)
    f_score = {nodo: float("inf") for fila in grid for nodo in fila}
    f_score[inicio] = h(inicio.get_pos(), fin.get_pos())

    open_set_hash = {inicio} 

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Obtener el nodo con el f_score más bajo
        actual = open_set.get()[2] 
        open_set_hash.remove(actual)

        if actual == fin:
            reconstruir_camino(came_from, fin, dibujar)
            fin.hacer_fin() 
            inicio.hacer_inicio() 
            return True

       
        for vecino in actual.vecinos:
            
            # CALCULO DE COSTO INTELIGENTE:
            if actual.fila != vecino.fila and actual.col != vecino.col:
                costo = 1.414  # Raíz cuadrada de 2
            else:
                costo = 1      # Movimiento recto
            
            # Usamos ese costo variable aquí:
            temp_g_score = g_score[actual] + costo

            if temp_g_score < g_score[vecino]:
                came_from[vecino] = actual
                g_score[vecino] = temp_g_score
                f_score[vecino] = temp_g_score + h(vecino.get_pos(), fin.get_pos())
                
                if vecino not in open_set_hash:
                    count += 1
                    open_set.put((f_score[vecino], count, vecino))
                    open_set_hash.add(vecino)
                    vecino.hacer_abierto()

        dibujar() 

        if actual != inicio:
            actual.hacer_cerrado() # Pintar de rojo (cerrado)

    return False # No se encontró un camino

# --- Funciones de Dibujo 

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
    y, x = pos 
    
   
    x_pixel, y_pixel = pos
    
    
    fila = x_pixel // ancho_nodo
    col = y_pixel // ancho_nodo
    
    if 0 <= fila < filas and 0 <= col < filas:
        return fila, col
    return None 

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
                continue 

            # --- Manejo de Clicks ---
            if pygame.mouse.get_pressed()[0]:  # Click izquierdo
                pos = pygame.mouse.get_pos()
                click_pos = obtener_click_pos(pos, FILAS, ancho)
                if not click_pos: continue 
                
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
                if event.key == pygame.K_SPACE and inicio and fin:
                    algoritmo_iniciado = True
                    for fila_grid in grid:
                        for nodo in fila_grid:
                            nodo.actualizar_vecinos(grid)
                    
                    algoritmo_a_estrella(
                        lambda: dibujar(ventana, grid, FILAS, ancho), # Función de dibujo
                        grid, 
                        inicio, 
                        fin
                    )

                if event.key == pygame.K_c:
                    grid = crear_grid(FILAS, ancho)
                    inicio = None
                    fin = None
                    algoritmo_iniciado = False


    pygame.quit()

if __name__ == "__main__":
    main(VENTANA, ANCHO_VENTANA)