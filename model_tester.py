import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np

model = tf.keras.models.load_model('./sign_language_model.h5')

camera = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands =  mp_hands.Hands(static_image_mode=True,
                        max_num_hands=2,
                        min_detection_confidence=0.3)

label_mapping = {0:"A", 1:"B", 2:"C", 3:"D", 4:"Eat", 5:"F", 6:"G", 7:"H", 8:"Hello", 9:"I", 10:"I love you", 11:"L", 12:"My", 13:"No", 14:"O",
                 15:"P", 16:"R", 17:"U", 18:"V", 19:"W", 20:"X", 21:"Y"}

expected_landmark_count = 21  # number of landmarks expected for one hand

while True:
    image_data = []
    image_x_values = []
    image_y_values = []

    captured_successfully, frame = camera.read()
    if not captured_successfully:
        continue

    height, width, placeholder = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if (len(hand_landmarks.landmark) == expected_landmark_count):
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                image_x_values = [landmark.x for landmark in hand_landmarks.landmark]
                image_y_values = [landmark.y for landmark in hand_landmarks.landmark]

                for i in range(expected_landmark_count):
                    image_data.append(image_x_values[i] - min(image_x_values))
                    image_data.append(image_y_values[i] - min(image_y_values))
                    
                if len(image_data) == 42:
                    x1 = int(min(image_x_values) * width) - 10
                    y1 = int(min(image_y_values) * height) - 10

                    x2 = int(max(image_x_values) * width) - 10
                    y2 = int(max(image_y_values) * height) - 10

                    frame_data = np.asarray(image_data).reshape(1, -1) # converting the list to a numpy array and reshape it to match model's input shape

                    prediction = model.predict(frame_data) # make a prediction using the model

                    # decoding the prediction
                    predicted_class_index = np.argmax(prediction)
                    predicted_phrase = label_mapping[predicted_class_index]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(img = frame, text = predicted_phrase, org = (x1, y1 - 10), 
                                fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1.3, color = (0, 0, 255), thickness = 3, lineType = cv2.LINE_AA)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"): # press 'q' on keyboard to quit the program
        break

camera.release()
cv2.destroyAllWindows()
