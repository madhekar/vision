import os
import cv2
from insightface.app import FaceAnalysis

def clip_faces(image_path, output_dir="cropped_faces"):
    """
    Scans an image for faces using deep learning and saves each cropped face.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the FaceAnalysis app (downloads the model on first run)
    # det_size controls the input resolution for detection
    app = FaceAnalysis(allowed_modules=['detection'])
    app.prepare(ctx_id=0, det_size=(640, 640)) # Set ctx_id=-1 if you don't have a GPU
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not look up or load image at {image_path}")
        return

    # Detect faces
    faces = app.get(img)
    print(f"Found {len(faces)} face(s) in the image.")

    # Loop through detected faces and crop them
    for i, face in enumerate(faces):
        # Get bounding box coordinates (bbox format: [x_min, y_min, x_max, y_max])
        bbox = face.bbox.astype(int)
        x_min, y_min, x_max, y_max = bbox
        
        # Ensure coordinates are within image boundaries
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(img.shape[1], x_max), min(img.shape[0], y_max)
        
        # Crop the face from the original image
        cropped_face = img[y_min:y_max, x_min:x_max]
        
        # Save the cropped face
        output_filename = os.path.join(output_dir, f"face_{i+1}.jpg")
        cv2.imwrite(output_filename, cropped_face)
        print(f"Saved: {output_filename}")

if __name__ == "__main__":
    # Replace with your image file path
    target_image = "your_image.jpg" 
    clip_faces(target_image)
