import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
from deepface import DeepFace
from ast import literal_eval

p = ['Anjali', 'Asha', 'Bhalchandra', 'Bhiman', 'Chandrakant', 'Esha', 'Kumar', 'Sachi', 'Sanvi', 'Shibangi', 'Shoma']

# Load the trained SVM model and label encoder
with open("svm_recognizer.pkl", "rb") as f:
    svm_classifier = pickle.load(f)

with open("le_recognizer.pkl", "rb") as f:
    le = pickle.load(f)

# Initialize InsightFace model
app = FaceAnalysis(name="buffalo_l")
#app = FaceAnalysis(allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=-1, det_size=(640, 640))

# Load a new image for recognition
new_image_path = "/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/b6f657c7-7b7f-5415-82b7-e005846a6ef5/f6b572ec-a56f-4d66-b900-fa61b48ce005.jpg"
#"/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/2a98fafb-a921-519f-8561-ed25ccd997de/5e144618-aea4-4365-95ad-375ae00a1133.jpg"
new_img = cv2.imread(new_image_path)
people = []
# Get face embedding
faces = app.get(new_img)
if faces:
    print(f"Detected {len(faces)} face(s).")
    for i, face in enumerate(faces):
        person = {}
        print(f"\nFace {i + 1}:")

        kps = face.kps.astype(int)
        # print(f"  Keypoints: {kps}")
        # Draw keypoints on the image
        for kp in kps:
            cv2.circle(new_img, tuple(kp), 1, (0, 0, 255), 2)
        # print('--->',face)
        # Gender and Age (if available in the model)
        if "gender" in face and "age" in face:
            gender = "Male" if face.gender == 1 else "Female"
            print(f"  Gender: {gender}, Age: {face.age}")
            person['age'] = face.age
            person['gender'] = gender

        
        new_embedding = faces[i].embedding

        # Predict the identity using the trained SVM
        prediction = svm_classifier.predict([new_embedding])[0]
        class_probabilities = svm_classifier.predict_proba([new_embedding])[0]

        # Get the predicted class label
        predicted_person = le.inverse_transform([prediction])[0]
        confidence = np.max(class_probabilities)
        if confidence < 0.6:
            predicted_person = "unk"
        else:
            predicted_person = p[predicted_person]

        person['name'] = predicted_person

        x1, y1, x2, y2 = face.bbox.astype(int)
        cropped_face = new_img[y1:y2, x1:x2]

        em = DeepFace.analyze(
        cropped_face,
        actions=['emotion'],
        #detector_backend='retinaface',
        enforce_detection=False
        )
        person["emotion"] = em[0]["dominant_emotion"]
        # print(f"Recognized as: {predicted_person} with confidence: {confidence:.2f}")


        # Display the image with the prediction
        bbox = faces[i].bbox.astype(int)
        cv2.rectangle(new_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        text = f"{predicted_person}: {confidence:.2f}"
        cv2.putText(
            new_img,
            str(person),
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )
        people.append(person)
    print(people)
    cv2.imshow("Recognized Face", new_img)
    if cv2.waitKey(100000) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
else:
    print("No face detected in the image.")
