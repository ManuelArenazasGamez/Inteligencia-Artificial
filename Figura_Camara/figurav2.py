import cv2
import mediapipe as mp
import math
import numpy as np  # Necesario para crear el array de puntos para el polígono

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# max_num_hands=2 asegura la detección de ambas manos
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)

# ID del Dedo Índice (Punta) en el modelo de MediaPipe (Landmark 8)
INDEX_FINGER_TIP_ID = 8

# Captura de video
cap = cv2.VideoCapture(0)

# Variables para almacenar las coordenadas de los dedos
p_idx_right = None
p_idx_left = None

# --- Ya no necesitamos los parámetros del rectángulo fijo ---

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede recibir el frame. Saliendo ...")
        break

    # Voltear la imagen horizontalmente y obtener dimensiones
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convertir a RGB y procesar
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Reiniciar puntos
    p_idx_right = None
    p_idx_left = None
    
    # 1. Identificar Puntos Clave
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            
            # Determinar si es mano derecha o izquierda (importante al voltear la imagen)
            label = handedness.classification[0].label
            
            # Obtener las coordenadas en píxeles de la punta del dedo índice (Landmark 8)
            landmark = hand_landmarks.landmark[INDEX_FINGER_TIP_ID]
            x, y = int(landmark.x * w), int(landmark.y * h)
            
            # Almacenar el punto
            if label == 'Right':
                # Mano derecha del usuario
                p_idx_right = (x, y)
            elif label == 'Left':
                # Mano izquierda del usuario
                p_idx_left = (x, y)
            
            # Dibujar las conexiones de la mano (para visualizar la mano)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Dibujar círculos en los índices
            cv2.circle(frame, (x, y), 8, (255, 0, 255), -1)


    # 2. Calcular y Dibujar el Cuadrado Rotado
    
    if p_idx_right and p_idx_left:
        
        # p1 y p2 son el primer lado del cuadrado
        p1 = p_idx_right
        p2 = p_idx_left
        x1, y1 = p1
        x2, y2 = p2
        
        # Calcular el vector v = p2 - p1
        dx = x2 - x1
        dy = y2 - y1
        
        # Calcular el vector perpendicular (rotado 90 grados)
        # v_perp = (-dy, dx)
        
        # Calcular los puntos p3 y p4
        # p3 = p2 + v_perp
        # p4 = p1 + v_perp
        p3 = (int(x2 - dy), int(y2 + dx))
        p4 = (int(x1 - dy), int(y1 + dx))
        
        # --- Dibujar el Cuadrado Rotado ---
        
        # Crear un array de NumPy con los 4 puntos del cuadrado
        pts = np.array([p1, p2, p3, p4], dtype=np.int32)
        
        # Dibujar el polígono (cuadrado) relleno
        cv2.fillPoly(frame, [pts], (255, 0, 0)) # Relleno azul
        
        # Opcional: Dibujar el contorno del cuadrado
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=3)


    # Mostrar la imagen
    cv2.imshow("Cuadrado Rotable con Manos", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()