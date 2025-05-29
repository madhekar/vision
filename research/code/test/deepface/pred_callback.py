import tensorflow as tf
from deepface import DeepFace

class DeepFaceValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, deepface_model_name="VGG-Face"):  # Example model name
        super(DeepFaceValidationCallback, self).__init__()
        self.validation_data = validation_data
        self.deepface_model = DeepFace.build_model(deepface_model_name) #load deepface model

    def on_epoch_end(self, epoch, logs=None):
        # Load images from validation_data
        img1_path = self.validation_data[0]
        img2_path = self.validation_data[1]


        # Run DeepFace verification
        result = DeepFace.verify(img1_path=img1_path, img2_path=img2_path, model=self.deepface_model)

        # Log verification result
        print(f"\nEpoch {epoch + 1}: DeepFace verification result: {result['verified']}")

# Example usage
# assuming you have paths to validation image pairs:
validation_data_pairs = [("img1.jpg","img2.jpg"), ("img3.jpg", "img4.jpg")]

deepface_callback = DeepFaceValidationCallback(validation_data=validation_data_pairs)

model = tf.keras.models.Sequential([
    # your keras model layers
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Train with the callback (assuming validation_data is a list of paths or something similar)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, callbacks=[deepface_callback])
