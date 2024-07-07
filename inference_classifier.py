import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

video_path = './video/Y2meta.app-deletreo en lengua de señas-(480p).mp4'
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
               6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
               18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
               24: 'Y', 25: 'A'}

signing_in_progress = False
sign_start_time = 0
sign_sequence = []

detected_text = ""

def recognize_sign(hand_landmarks):
    data_aux = []
    x_ = []
    y_ = []

    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        x_.append(x)
        y_.append(y)

    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        data_aux.append(x - min(x_))
        data_aux.append(y - min(y_))

    return data_aux

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            
            data_aux = recognize_sign(hand_landmarks)
            sign_sequence.append(data_aux)

       
        if not signing_in_progress:
            signing_in_progress = True
            sign_start_time = time.time()
        else:
            if time.time() - sign_start_time > 4.0: 
                sign_data = np.array(sign_sequence)
                sign_data = np.mean(sign_data, axis=0)  

                
                prediction = model.predict([sign_data])
                predicted_character = labels_dict[int(prediction[0])]

                detected_text += predicted_character

                sign_sequence = []
                signing_in_progress = False

    cv2.putText(frame, detected_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Sign Language to Text', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Texto detectado:", detected_text)
