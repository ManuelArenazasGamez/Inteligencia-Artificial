import cv2
import mediapipe as mp
import math


MODO_CALIBRACION = False  


# --- Inicialización de MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(234, 255, 233))

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


P_OJO_IZQ_INTERNO = 133
P_OJO_DER_INTERNO = 362

P_BOCA_IZQ = 291
P_BOCA_DER = 61
P_BOCA_INFERIOR_CENTRO = 17

# Cejas (NUEVA LÓGICA DE ENOJO: ALTURA)
P_CEJA_IZQ_SUP = 336 # Punto superior de la ceja izq
P_OJO_IZQ_SUP = 386  # Punto superior del párpado izq

def calcular_distancia(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

cap = cv2.VideoCapture(0)

# --- UMBRALES (AJUSTADOS A  DATOS PERSONALES) ---

# FELIZ: (Tu Neutral 1.42, Tu Feliz 1.80)
UMBRAL_FELIZ = 1.60 

# TRISTE: (Tu Neutral -0.29, Tu Triste -0.20)
UMBRAL_TRISTE = -0.25

# ENOJADO: (Tu Neutral 0.82, Tu Enojado 0.69)
UMBRAL_ENOJO = 0.76  


while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    emocion = "Neutral"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            lm = face_landmarks.landmark
            def get_px(landmark_id):
                return (int(lm[landmark_id].x * w), int(lm[landmark_id].y * h))

            # Dibujar la malla
            mp_drawing.draw_landmarks(
                image=frame, landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            # --- 1. Extraer Puntos ---
            p_ojo_izq = get_px(P_OJO_IZQ_INTERNO)
            p_ojo_der = get_px(P_OJO_DER_INTERNO)
            p_boca_izq = get_px(P_BOCA_IZQ)
            p_boca_der = get_px(P_BOCA_DER)
            p_boca_inf_centro = get_px(P_BOCA_INFERIOR_CENTRO)
            p_ceja_izq_sup = get_px(P_CEJA_IZQ_SUP)
            p_ojo_izq_sup = get_px(P_OJO_IZQ_SUP)

            # --- 2. Calcular Ratios ---
            dist_ojos = calcular_distancia(p_ojo_izq, p_ojo_der)
            if dist_ojos == 0: continue

            # Ratio FELIZ (Ancho de la boca)
            dist_boca = calcular_distancia(p_boca_izq, p_boca_der)
            ratio_feliz = dist_boca / dist_ojos
            
            # Ratio TRISTE (Comisuras bajas)
            y_corners_avg = (p_boca_izq[1] + p_boca_der[1]) / 2
            y_lip_bottom = p_boca_inf_centro[1]
            dist_frown = y_corners_avg - y_lip_bottom
            ratio_triste = dist_frown / dist_ojos

            # Ratio ENOJO (NUEVA LÓGICA: Altura de la ceja)
            dist_ceja_ojo = calcular_distancia(p_ceja_izq_sup, p_ojo_izq_sup)
            ratio_ceja_altura = dist_ceja_ojo / dist_ojos

            # --- 3. Lógica de Decisión ---
            if MODO_CALIBRACION:
                cv2.putText(frame, f"FELIZ: {ratio_feliz:.2f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"TRISTE: {ratio_triste:.2f}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"CEJA_ALTURA: {ratio_ceja_altura:.2f}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            else:
                # Modo Detección
                # El orden importa: revisamos las emociones más "extremas" primero
                if ratio_feliz > UMBRAL_FELIZ:
                    emocion = "Feliz"
                elif ratio_triste > UMBRAL_TRISTE:
                    emocion = "Triste"
                elif ratio_ceja_altura < UMBRAL_ENOJO: # Nota: es MENOR QUE
                    emocion = "Enojado"
                else:
                    emocion = "Neutral"
                
                cv2.putText(frame, emocion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Detector de Emociones Final', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()