import os
import cv2
import joblib
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from insightface.app import FaceAnalysis
from utils.config_util import config

def init_model():
    # Initialize InsightFace model
    app = FaceAnalysis(allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(640, 640))  # Use ctx_id=0 for GPU, -1 for CPU
    return app


def train_model(app, img_dataset_path, model_path, model_file, label_path, label_file):
    # Prepare lists for embeddings and labels
    embeddings_list = []
    labels_list = []
    class_names = []
    image_path_list = []

    # Loop through the dataset directory to get embeddings
    for class_label, person_name in enumerate(sorted(os.listdir(img_dataset_path))):
        person_path = os.path.join(img_dataset_path, person_name)
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

    # Save the trained model and label encoder for future use
    print("Saving model and label encoder...")
    if not os.path.exists(model_path):
       os.makedirs(model_path)
    joblib.dump(svm_classifier, filename=os.path.join(model_path, model_file))

    le = LabelEncoder()
    le.fit_transform(class_names)
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    joblib.dump(le, filename=os.path.join(label_path, label_file))    
    print("Model saved successfully.")

def execute():
    (
        faces_dir,
        input_image_path,
        class_embeddings_folder,
        class_embeddings,
        label_encoder_path,
        label_encoder,
        faces_svc_path,
        faces_svc,
        faces_of_people_parquet_path, 
        faces_of_people_parquet
    ) = config.faces_config_load()

    # label_encoder_store = os.path.join(label_encoder_path, label_encoder)
    # svc_model_store = os.path.join(faces_svc_path, faces_svc)

    # init train model
    app = init_model()

    # train model using faces dataset
    train_model(
        app, faces_dir, faces_svc_path, faces_svc, label_encoder_path, label_encoder
    )

if __name__=='__main__':    
    execute()