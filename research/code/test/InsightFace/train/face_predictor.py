import cv2
import pickle
import pandas as pd
import numpy as np
from insightface.app import FaceAnalysis
from deepface import DeepFace
from ast import literal_eval


p = ['Anjali', 'Asha', 'Bhalchandra', 'Bhiman', 'Chandrakant', 'Esha', 'Kumar', 'Sachi', 'Sanvi', 'Shibangi', 'Shoma']
template_1 = "A face at cocordinates {loc} of {name} appreas to be {age} years {gender} and expressing {emotion} emotion. "
#template_2 = "Another face at cocordinates {loc} of {name} appreas to be {age} years {gender} and expressing {emotion} emotion. "

# Load the trained SVM model and label encoder
with open("svm_recognizer.pkl", "rb") as f:
    svm_classifier = pickle.load(f)

with open("le_recognizer.pkl", "rb") as f:
    le = pickle.load(f)

def determine_derived(age, gender):
    txt = ""
    if age < 21:
        if gender == 'Female':
           txt = "girl"
        else:
            txt = "boy"
    else:
        if gender == 'Female':
           txt = "woman"
        else:
            txt = "man"   
    return txt            

def create_partial_llm_prompt(df):
    llm_prompt = ""
    for index, row in df.iterrows():
        if row['name'] != "unknown":
            llm_prompt += f"A face at cocordinates {row["loc"]} of {row["name"]} appreas to be {row["age"]} years {gender} and expressing {row["emotion"]} emotion. "


# Initialize InsightFace model
app = FaceAnalysis(name="buffalo_l")
#app = FaceAnalysis(allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=-1, det_size=(640, 640))

# Load a new image for recognition
new_image_path = "/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/b6f657c7-7b7f-5415-82b7-e005846a6ef5/7e9a4cc3-b380-40ff-a391-8bf596f8cd27.jpg"
#"/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/b6f657c7-7b7f-5415-82b7-e005846a6ef5/f6b572ec-a56f-4d66-b900-fa61b48ce005.jpg"
#"/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/2a98fafb-a921-519f-8561-ed25ccd997de/5e144618-aea4-4365-95ad-375ae00a1133.jpg"
new_img = cv2.imread(new_image_path)
people = []
# Get face embedding
faces = app.get(new_img)
if faces:
    print(f"Detected {len(faces)} face(s).")
    for i, face in enumerate(faces):
        person = {}


        if "gender" in face and "age" in face:
            gender = "Male" if face.gender == 1 else "Female"
            # detect age and gender for the face
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
            predicted_person = "unknown"
        else:
            predicted_person = p[predicted_person]

        # name for face in the image
        person['name'] = predicted_person

        x1, y1, x2, y2 = face.bbox.astype(int)
        cropped_face = new_img[y1:y2, x1:x2]

        #face location in an image
        person['loc'] = (x1, y1)

        em = DeepFace.analyze(cropped_face,actions=['emotion'],enforce_detection=False)
        # face emotion 
        person["emotion"] = em[0]["dominant_emotion"]

        # Display the image with the prediction
        cv2.rectangle(new_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{predicted_person}: {confidence:.2f}"
        cv2.putText(
            new_img,
            str(person),
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )
        people.append(person)
    print(people)
    df = pd.DataFrame(people)
    print(df.head())
    cv2.imshow("Recognized Face", new_img)
    if cv2.waitKey(100000) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
else:
    print("No face detected in the image.")
