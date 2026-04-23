# importing libraries
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# loading trained model
model = load_model("model/emotion_model.keras")

# loading face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# emotion labels
emotion_labels = ["angry","disgust","fear","happy","neutral","sad","surprise"]

# starting webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # converting to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecting faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # extracting face
        roi = gray[y:y+h, x:x+w]

        # resizing
        roi = cv2.resize(roi, (48,48))

        # normalizing
        roi = roi / 255.0

        # reshaping
        roi = np.reshape(roi, (1,48,48,1))

        # predicting emotion
        prediction = model.predict(roi, verbose=0)
        label = emotion_labels[np.argmax(prediction)]

        # drawing rectangle
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        # putting text
        cv2.putText(frame, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,255,0), 2)

    # showing output
    cv2.imshow("Emotion Tracker", frame)

    # exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()