# in this file, we will be using OpenCV to take pictures and create our dataset for the different signs

import os
import cv2

images_folder = "./images"

# only going to be using these letters/phases as many of the letters look very similar which
# will causes issues with the image classification
classes = ["a", "b", "c", "d", "f", "g", "h", "i", "l", "o", "p", "r", "u", "v", "w", "x", "y", "z", 
           "my", "hello", "no", "i-love-you", "eat"]

size_of_dataset = 100 # 100 images for each letter/phrase

camera = cv2.VideoCapture(0)

for phrase in classes:
    if not os.path.exists(os.path.join(images_folder, phrase)):
        os.makedirs(os.path.join(images_folder, phrase))

    print(f"currently getting images for '{phrase}'")

    while True:
        captured_successfully, frame = camera.read()
        cv2.imshow("frame", frame)
        if (cv2.waitKey(25) == ord("c")): # press 'c' on keyboard to begin getting images for current phrase
            break
    
    num_of_pics = 0

    while num_of_pics < size_of_dataset:
        captured_successfully, frame = camera.read()
        cv2.imshow("frame", frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(images_folder, phrase, f"{str(num_of_pics)}.jpg"), frame)
        num_of_pics += 1

camera.release()
cv2.destroyAllWindows()