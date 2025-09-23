import insightface
import numpy as np
import cv2
from sklearn.svm import SVC  # Example classifier

# Load InsightFace's face detection and landmark model
app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=-1, det_size=(640, 640))

# Load a pre-trained emotion classification model (this would be custom-built)
# For demonstration, let's assume a simple SVM trained on landmark features
# In a real scenario, this would be a deep learning model
emotion_classifier = SVC(kernel="linear")
# ... (Load trained emotion_classifier from a saved model)

# Load an image
img = cv2.imread("/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/2a98fafb-a921-519f-8561-ed25ccd997de/e8828925-b35c-4779-a62e-1adcb11a156c.jpg")

# Detect faces and get landmarks
faces = app.get(img)

for face in faces:
    landmarks = face.kps
    #landmarks = face.landmark_2d_106.flatten()  # Flatten landmarks for classifier input
    print(landmarks)
    # Predict emotion using the custom classifier
    predicted_emotion = emotion_classifier.predict([landmarks])[0]

    print(f"Detected Emotion: {predicted_emotion}")
