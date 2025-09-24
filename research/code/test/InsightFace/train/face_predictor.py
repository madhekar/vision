import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# Load the trained SVM model and label encoder
with open("svm_recognizer.pkl", "rb") as f:
    svm_classifier = pickle.load(f)

with open("le_recognizer.pkl", "rb") as f:
    le = pickle.load(f)

# Initialize InsightFace model
app = FaceAnalysis(allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=-1, det_size=(640, 640))

# Load a new image for recognition
new_image_path = "/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/2a98fafb-a921-519f-8561-ed25ccd997de/5e144618-aea4-4365-95ad-375ae00a1133.jpg"
new_img = cv2.imread(new_image_path)

# Get face embedding
faces = app.get(new_img)
if faces:
    print(f"Detected {len(faces)} face(s).")
    for i, face in enumerate(faces):
            print(f"\nFace {i + 1}:")
            new_embedding = faces[i].embedding

            # Predict the identity using the trained SVM
            prediction = svm_classifier.predict([new_embedding])[0]
            class_probabilities = svm_classifier.predict_proba([new_embedding])[0]

            # Get the predicted class label
            predicted_person = le.inverse_transform([prediction])[0]
            confidence = np.max(class_probabilities)

            print(f"Recognized as: {predicted_person} with confidence: {confidence:.2f}")

            # Display the image with the prediction
            bbox = faces[i].bbox.astype(int)
            cv2.rectangle(new_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            text = f"{predicted_person}: {confidence:.2f}"
            cv2.putText(
                new_img,
                text,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )

    cv2.imshow("Recognized Face", new_img)
    if cv2.waitKey(10000) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
else:
    print("No face detected in the image.")
