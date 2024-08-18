import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import random
import time
import queue

model = tf.keras.models.load_model('./model/sign_language_model.h5')
camera = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)
current_target = None  # Initialize to prevent undefined errors
user_won = None

# model label mapping
label_mapping = {0:"A", 1:"B", 2:"C", 3:"D", 4:"Eat", 5:"F", 6:"G", 7:"H", 8:"Hello", 9:"I", 10:"I love you", 11:"L", 12:"My", 13:"No", 14:"O",
                 15:"P", 16:"R", 17:"U", 18:"V", 19:"W", 20:"X", 21:"Y"}
expected_landmark_count = 21
st.title("Sign Language Detection Test")
feedback_placeholder = st.empty()
frame_placeholder = st.empty()
recognized_signs_placeholder = st.empty()  # Define this at the start with other placeholders
timer_placeholder = st.empty()
start_test = st.button('Start Test')

# initialize the game state
game_duration = 30
sign_count = 4
recognized_signs = 0
game_start_time = None
test_running = None

if st.button("Go Back to Home"):
    st.markdown(f'<meta http-equiv="refresh" content="0; url=https://project-sign-88f00.web.app/">', unsafe_allow_html=True)

# Singular
def get_random_label():
    return random.choice(list(label_mapping.values()))

# Multiple
def get_random_labels(num_labels):  
    return random.sample(list(label_mapping.values()), num_labels)

# Setup a queue for the signs
def setup_sign_queue(signs):
    q = queue.Queue()
    for sign in signs:
        q.put(sign)
    return q


if start_test:
    user_won = False
    recognized_signs = 0
    game_start_time = time.time()
    test_running = True
    target_labels = get_random_labels(sign_count)
    signs_queue = setup_sign_queue(target_labels)
    current_target = signs_queue.get() if not signs_queue.empty() else None
    feedback_placeholder = st.empty()
    feedback_placeholder.info(f"Signs to perform: {', '.join(target_labels)}")

while True:

    if test_running and game_start_time:
        time_elapsed = time.time() - game_start_time
        remaining_time = game_duration - time_elapsed
        recognized_signs_placeholder.text(f"Recognized Signs: {recognized_signs}/{sign_count}")
        timer_placeholder.text(f"Time left: {int(remaining_time)} seconds")
        if remaining_time <= 0:
            feedback_placeholder.error("Time's up! Test over.")
            test_running = False
            game_start_time = None


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
                    if test_running:
                        # Check if user has matched the target label
                        if predicted_phrase == current_target:
                            recognized_signs += 1
                            feedback_placeholder.success(f"Correct! Recognized: {predicted_phrase}")
                            current_target = signs_queue.get() if not signs_queue.empty() else None
                            if current_target:
                                feedback_placeholder.info(f"Next sign to perform: {current_target}")
                            else:
                                feedback_placeholder.success("Congratulations! All signs recognized.")
                                test_running = False
                                game_start_time = None
                                break
                        else:
                            feedback_placeholder.warning(f"Try again! Expected: {current_target}, but got: {predicted_phrase}")

    # display the frame and score in streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)


camera.release()
cv2.destroyAllWindows()