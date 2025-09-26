import os
import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
from insightface.app import FaceAnalysis

# Initialize InsightFace model
app = FaceAnalysis(allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=-1, det_size=(640, 640))  # Use ctx_id=0 for GPU, -1 for CPU

# Prepare lists for embeddings and labels
embeddings_list = []
labels_list = []
class_names = []
image_path_list = []

# Loop through the dataset directory to get embeddings
dataset_path = "/home/madhekar/work/home-media-app/data/app-data/static-metadata/faces"
for class_label, person_name in enumerate(sorted(os.listdir(dataset_path))):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue

    class_names.append(person_name)

    for image_file in os.listdir(person_path):
        image_path = os.path.join(person_path, image_file)
        if not os.path.isfile(image_path):
            continue

        try:
            img = cv2.imread(image_path)
            if img is None:
                continue
            print('--->', image_path)
            # Get faces from the image
            faces = app.get(img)

            # Ensure a face is detected
            if faces:
                # The recognition model returns one embedding per face
                embedding = faces[0].embedding
                embeddings_list.append(embedding)
                labels_list.append(class_label)
                image_path_list.append(image_path)

        except Exception as e:
            print(f"Could not process image {image_path}: {e}")

# Convert lists to numpy arrays
X = np.array(embeddings_list)
y = np.array(labels_list)
print('--->', y)
print(f"Generated {len(X)} embeddings for training.")
print(f"The unique classes are: {class_names}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Initialize and train the SVM classifier
print("Training SVM classifier...")
svm_classifier = SVC(kernel="rbf", probability=True)
svm_classifier.fit(X_train, y_train)
print("Training complete.")

# Evaluate the classifier
y_pred = svm_classifier.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

'''

    """
    Label encoder
    """
    st.info('creating new labels for faces...')
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    ss.create_folder(label_encoder_path)
    joblib.dump(encoder, filename=os.path.join(label_encoder_path, label_encoder))

    """
    train SVC
    """
    model = SVC(kernel="rbf", probability=True)
    model.fit(embedded_x, y)
    ss.create_folder(faces_svc_path)
    joblib.dump(model, filename=os.path.join(faces_svc_path, faces_svc))

'''

# Save the trained model and label encoder for future use
print("Saving model and label encoder...")
with open("svm_recognizer.pkl", "wb") as f:
    pickle.dump(svm_classifier, f)

joblib.dump(svm_classifier, filename="faces_model_svc.joblib")


le = LabelEncoder()
print('--->', y)
le.fit_transform(class_names)
with open("le_recognizer.pkl", "wb") as f:
    pickle.dump(le, f)

joblib.dump(le, filename="faces_label_enc.joblib")    
print("Model saved successfully.")