import cv2
import mediapipe as mp
import math

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

# --- Parámetros del Rectángulo Fijo ---
RECT_HEIGHT = 40  # Altura fija del rectángulo
RECT_Y_POS = 30   # Posición Y (vertical) fija para el rectángulo (cerca de la parte superior)

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


    # 2. Calcular Distancia y 3. Dibujar Rectángulo Fijo
    
    if p_idx_right and p_idx_left:
        
        x1, y1 = p_idx_right
        x2, y2 = p_idx_left
        
        # Calcular la distancia euclidiana
        distance = math.hypot(x2 - x1, y2 - y1)
        
        # Opcional: Dibujar la línea entre los dedos para referencia
        cv2.line(frame, p_idx_right, p_idx_left, (0, 255, 255), 3)
        
        # --- Dibujar Rectángulo en Posición Fija ---
        
        rect_width = int(distance)
        
        # Centrar el rectángulo en X (centro de la pantalla)
        start_x = (w // 2) - (rect_width // 2)
        end_x = (w // 2) + (rect_width // 2)
        
        # Usar la posición Y fija
        start_y = RECT_Y_POS
        end_y = RECT_Y_POS + RECT_HEIGHT
        
        pt1 = (start_x, start_y)
        pt2 = (end_x, end_y)
        
        # Dibujar el rectángulo (RELLENO) para crear la barra de distancia
        cv2.rectangle(frame, pt1, pt2, (255, 0, 0), -1) # -1 para rellenar
        
        # Mostrar el valor de la distancia sobre el rectángulo
        text_pos = (start_x + 10, start_y + RECT_HEIGHT - 10)
        cv2.putText(frame, f'{int(distance)} px', text_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    # Mostrar la imagen
    cv2.imshow("Distancia Manos Rectangulo yeah xd", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()