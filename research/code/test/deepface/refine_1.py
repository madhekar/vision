from deepface import DeepFace
import cv2

# Load the image
img = cv2.imread("/home/madhekar/temp/Deepface_issues/IMG_8543.PNG")

# Detect faces using a chosen backend (e.g., 'retinaface')
# This will return a list of dictionaries, each containing face details and bounding box
detected_faces = DeepFace.extract_faces(img_path="image_with_multiple_faces.jpg", detector_backend='retinaface')

# Iterate through each detected face
for face_data in detected_faces:
    x, y, w, h = face_data['facial_area']['x'], face_data['facial_area']['y'], face_data['facial_area']['w'], face_data['facial_area']['h']
    
    # Crop the face from the original image
    cropped_face = img[y:y+h, x:x+w]
    
    # Analyze or process the individual cropped face
    # For example, analyze demographics:
    analysis_results = DeepFace.analyze(img_path=cropped_face, actions=['age', 'gender', 'emotion'], enforce_detection=False)
    print(analysis_results)
    
    # Or perform face recognition against a database:
    # results = DeepFace.find(img_path=cropped_face, db_path="my_face_database", enforce_detection=False)
    # print(results)