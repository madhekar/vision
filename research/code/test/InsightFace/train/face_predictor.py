import cv2
import pickle
from ast import literal_eval
import joblib
import pandas as pd
import numpy as np
from insightface.app import FaceAnalysis
from deepface import DeepFace
from ast import literal_eval


p = ['Anjali', 'Asha', 'Bhalchandra', 'Bhiman', 'Chandrakant', 'Esha', 'Kumar', 'Sachi', 'Sanvi', 'Shibangi', 'Shoma']

# Load the trained SVM model and label encoder
# with open("svm_recognizer.pkl", "rb") as f:
#     svm_classifier = pickle.load(f)

svm_classifier = joblib.load("faces_model_svc.joblib")

# with open("le_recognizer.pkl", "rb") as f:
#     le = pickle.load(f)

le = joblib.load("faces_label_enc.joblib")

def count_people(ld):
    cnt_man = 0
    cnt_woman = 0
    cnt_boy = 0
    cnt_girl = 0

    agg_person = []
    for d in ld:
        if d["name"] == "unknown":
            if d["cnoun"] == "man":
                cnt_man += 1
            if d["cnoun"] == "woman":
                cnt_woman += 1
            if d["cnoun"] == "boy":
                cnt_boy += 1
            if d["cnoun"] == "girl":
                cnt_girl += 1
        else:
            agg_person.append({"type": "known", "name": d["name"], "cnoun": d["cnoun"],"emotion": d["emotion"], "loc": d["loc"]})

    agg_person.append({"type": "unknown","cman": cnt_man, "cwoman": cnt_woman,"cboy": cnt_boy,"cgirl": cnt_girl})

    return agg_person


def create_partial_prompt(agg):
    txt = ""
    for d in agg:
        if d["type"] == "known":
            s = f'Face at coordinates {d["loc"]} is of "{d["name"]}", a "{d["cnoun"]}" expressing "{d["emotion"]}" emotion. '
            txt += s
        if d["type"] == "unknown":
            txt += "And "
            if d["cman"] > 0:
                if d["cman"] > 1:
                    s = f" {d['cman']} men  "
                else:
                    s = " one  man  "
                txt += s

            if d["cwoman"] > 0:
                if d["cwoman"] > 1:
                    s = f" {d['cwoman']}  women  "
                else:
                    s = " one  woman  "
                txt += s

            if d["cboy"] > 0:
                if d["cboy"] > 1:
                    s = f" {d['cboy']} boys  "
                else:
                    s = " one  boy  "
                txt += s

            if d["cgirl"] > 0:
                if d["cgirl"] > 1:
                    s = f" {d['cgirl']}  girls  "
                else:
                    s = " one girl  "
                txt += s 
            txt += "in the image."
    return txt

''' 
Initialize InsightFace model
'''
def init_predictor():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app

'''
predict known and unknown faces
'''
def predic_img_faces(app, new_image_path):

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
                age = face.age - 10
                person['age'] = age
                person['gender'] = gender
                if age > 21:
                    if gender == "Female":
                        person["cnoun"] = "woman"
                    else:
                        person["cnoun"] = "man"  
                else:
                    if gender == "Female":
                        person["cnoun"] = "girl"
                    else:
                        person["cnoun"] = "boy"    


            new_embedding = faces[i].embedding

            # Predict the identity using the trained SVM
            prediction = svm_classifier.predict([new_embedding])[0]
            class_probabilities = svm_classifier.predict_proba([new_embedding])[0]
            
            print('--->',prediction, '::',le.inverse_transform([prediction]))
            # Get the predicted class label
            predicted_person = le.inverse_transform([prediction])[0]
            #literal_eval(str(le.inverse_transform([prediction])[0]).strip())
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
                str(text),
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )
            people.append(person)
        print(people)

        agg_cnt = count_people(people)

        llm_partial_pmt = create_partial_prompt(agg_cnt)

        print(llm_partial_pmt)

        cv2.imshow("Recognized Face", new_img)
        if cv2.waitKey(70000) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
    else:
        print("No face detected in the image.")


if __name__=='__main__':
    # Load a new image for recognition
    img = "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/2596441a-e02f-588c-8df4-dc66a133fc99/af23fb36-9e89-4b81-9990-020b02fe1056.jpg"
    # "/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/b6f657c7-7b7f-5415-82b7-e005846a6ef5/f6b572ec-a56f-4d66-b900-fa61b48ce005.jpg"
    # "/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/b6f657c7-7b7f-5415-82b7-e005846a6ef5/7e9a4cc3-b380-40ff-a391-8bf596f8cd27.jpg"
    # "/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/2a98fafb-a921-519f-8561-ed25ccd997de/5e144618-aea4-4365-95ad-375ae00a1133.jpg"

    app = init_predictor()

    predic_img_faces(app, img)