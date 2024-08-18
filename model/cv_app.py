import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import random
import time

model = tf.keras.models.load_model('./model/sign_language_model.h5')

# attempt to open the camera
camera_index = None
for i in range(5):
    camera = cv2.VideoCapture(i)
    if camera.isOpened():
        camera_index = i
        camera.release()
        break

if camera_index is None:
    st.write("Unable to access the camera. Please make sure a camera is connected and accessible")
    st.stop()

camera = cv2.VideoCapture(camera_index)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

# model label mapping
label_mapping = {0:"A", 1:"B", 2:"C", 3:"D", 4:"Eat", 5:"F", 6:"G", 7:"H", 8:"Hello", 9:"I", 10:"I love you", 11:"L", 12:"My", 13:"No", 14:"O",
                 15:"P", 16:"R", 17:"U", 18:"V", 19:"W", 20:"X", 21:"Y"}

expected_landmark_count = 21 # number of landmarks expected for one hand

st.title("Sign Language Detection")

frame_placeholder = st.empty()
feedback_placeholder = st.empty()
score_placeholder = st.empty()
timer_placeholder = st.empty()

if st.button("Go Back to Home"):
    st.markdown(f'<meta http-equiv="refresh" content="0; url=https://project-sign-88f00.web.app/">', unsafe_allow_html=True)

# initialize the game state
game_start_time = None
target_label = None
user_won = None
score = 0
game_duration = 10
display_result_time = None

def get_random_label():
    return random.choice(list(label_mapping.values()))

while True:
    current_time = time.time()

    if display_result_time and ((current_time - display_result_time) < 5):
        if game_start_time is not None: # keep showing "You Win!" or "You Lose!" message for 5 seconds
            remaining_time = max(0, game_duration - (display_result_time - game_start_time))
            timer_placeholder.text(f"Time left: {int(remaining_time)} seconds")
    else:
        if (game_start_time is None) or ((current_time - game_start_time) > game_duration):
            if (user_won is None) and (game_start_time is not None):  # time ran out and the user didn't win
                if score - 5 < 0:
                    score = 0
                else:
                    score -= 5
                user_won = False
                display_result_time = current_time
            else:
                # start a new game
                target_label = get_random_label()
                game_start_time = current_time
                user_won = None
                feedback_placeholder.text(f"Show this sign: {target_label}")
                display_result_time = None

        if game_start_time is not None:
            remaining_time = max(0, game_duration - (current_time - game_start_time))
            timer_placeholder.text(f"Time left: {int(remaining_time)} seconds")

        image_data = []
        image_x_values = []
        image_y_values = []

        captured_successfully, frame = camera.read()
        if not captured_successfully:
            st.write("The video capture has ended")
            break

        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            
            # predicting label from frame
            for hand_landmarks in results.multi_hand_landmarks:
                if len(hand_landmarks.landmark) == expected_landmark_count:
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

                        frame_data = np.asarray(image_data).reshape(1, -1)

                        prediction = model.predict(frame_data)

                        predicted_class_index = np.argmax(prediction)
                        predicted_phrase = label_mapping[predicted_class_index]

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(frame, predicted_phrase, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

                        # Check if user has matched the target label
                        if predicted_phrase == target_label:
                            user_won = True
                            display_result_time = current_time  # Start displaying "You Win!" message
                            score += 5
                            break

        if user_won is not None:
            if user_won:
                # display the "You Win!" message on the frame
                cv2.rectangle(frame, (50, 50), (width - 50, height - 50), (0, 255, 0), -1)
                cv2.putText(frame, "You Win!", (width // 2 - 100, height // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
                # reset the game after 5 seconds
                if (current_time - display_result_time) >= 5:
                    game_start_time = None
                    user_won = None
                    display_result_time = None
            else:
                # display the "You Lose!" message on the frame
                cv2.rectangle(frame, (50, 50), (width - 50, height - 50), (0, 0, 255), -1)
                cv2.putText(frame, "You Lose!", (width // 2 - 120, height // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
                if (current_time - display_result_time) >= 5: # reset the game after 5 seconds
                    game_start_time = None
                    user_won = None
                    display_result_time = None

    # display the frame and score in streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
    score_placeholder.text(f"Score: {score}")
    if game_start_time is not None:
        timer_placeholder.text(f"Time left: {int(remaining_time)} seconds")

camera.release()
cv2.destroyAllWindows()