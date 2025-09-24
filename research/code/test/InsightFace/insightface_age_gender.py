import cv2
import numpy as np
import insightface
from deepface import DeepFace
from insightface.app import FaceAnalysis

# Initialize the FaceAnalysis app with the desired model (buffalo_l is often the default)
# You can explicitly specify the model if needed: app = FaceAnalysis(name='buffalo_l')
app = FaceAnalysis()

# Prepare the model for inference. ctx_id=-1 uses CPU, ctx_id=0 uses GPU if available.
app.prepare(ctx_id=-1, det_size=(640, 640))

# Load an image
image_path = "/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/2a98fafb-a921-519f-8561-ed25ccd997de/5e144618-aea4-4365-95ad-375ae00a1133.jpg"
#"/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/2a98fafb-a921-519f-8561-ed25ccd997de/fce7616e-d485-403b-ba29-e33d5b80df09.jpg"
#"/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/b6f657c7-7b7f-5415-82b7-e005846a6ef5/7e9a4cc3-b380-40ff-a391-8bf596f8cd27.jpg"
#"/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/2a98fafb-a921-519f-8561-ed25ccd997de/e8828925-b35c-4779-a62e-1adcb11a156c.jpg"  # Replace with your image path
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # Perform face analysis
    faces = app.get(img)

    if len(faces) == 0:
        print("No faces detected in the image.")
    else:
        people = []

        print(f"Detected {len(faces)} face(s).")
        for i, face in enumerate(faces):
            print(f"\nFace {i + 1}:")
            person = {}
            # Bounding box
            bbox = face.bbox.astype(int)
            #print(f"  Bounding Box: {bbox}")
            # Draw bounding box on the image
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            person['loc'] = bbox
            # Keypoints (landmarks)
            kps = face.kps.astype(int)
            #print(f"  Keypoints: {kps}")
            # Draw keypoints on the image
            for kp in kps:
                cv2.circle(img, tuple(kp), 1, (0, 0, 255), 2)
            #print('--->',face)
            # Gender and Age (if available in the model)
            if "gender" in face and "age" in face:
                gender = "Male" if face.gender == 1 else "Female"
                print(f"  Gender: {gender}, Age: {face.age}")
                person['age'] = face.age
                person['gender'] = gender

            # Face embeddings (for face recognition/comparison)
            embedding = face.normed_embedding
            #print(f"  Embedding shape: {embedding.shape}")

            ##
            
            x1, y1, x2, y2 = face.bbox.astype(int)
            cropped_face = img[y1:y2, x1:x2]

            em = DeepFace.analyze(
            cropped_face,
            actions=['emotion'],
            #detector_backend='retinaface',
            enforce_detection=False
            )
            #people.append(em[0]['dominant_emotion'])
            #person += f" emotion: {em[0]['dominant_emotion']}"
            #print('***', em)
            person["emotion"] = em[0]["dominant_emotion"]
            people.append(person)
        print('--->', people)
        #Display the image with detections
        cv2.imshow("Detected Faces", img)
        if cv2.waitKey(10000) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
