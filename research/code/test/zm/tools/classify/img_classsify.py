import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
#import matplotlib
#matplotlib.use("Qt5Agg")
#print(matplotlib.get_backend())
#import matplotlib.pyplot as plt
import numpy as np
import os

"""
Epoch 100/100
11/11 ━━━━━━━━━━━━━━━━━━━━ 27s 2s/step - accuracy: 0.9764 - loss: 0.0438 - val_accuracy: 0.8333 - val_loss: 0.1350
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 543ms/step
Image: /home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/76b281d8-f830-4f24-9b45-ec02e88b52ac.jpg
Predicted class: people with confidence 0.91
predicted: people actual: people
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
Image: /home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/8109e957-ce86-4283-96aa-11c55d2fba62-1.jpg
Predicted class: people with confidence 0.60
predicted: people actual: people
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
Image: /home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/129c39e4-e7bf-4cfd-9bdd-0eb541bf9a60.jpg
Predicted class: scenic with confidence 0.99
predicted: scenic actual: scenic
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step
Image: /home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/679e1665-6c43-4018-8a8f-ffb2f33739b6-1.jpg
Predicted class: scenic with confidence 0.74
predicted: scenic actual: scenic
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
Image: /home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/7f5a9544-f625-4bbf-ad25-e0b874a4e5fd-1.jpg
Predicted class: document with confidence 0.97
predicted: document actual: document
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step
Image: /home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/107e99ff-289f-4d92-b2e6-c4943dccbccd.jpg
Predicted class: document with confidence 0.94
predicted: document actual: document
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
Image: /home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/91f2dc11-cffd-4d08-8575-7d437ef31774.jpg
Predicted class: scenic with confidence 1.00
predicted: scenic actual: scenic
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
Image: /home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/121b15b7-81b3-4792-b654-c95aa2ee4b49-1.jpg
Predicted class: scenic with confidence 0.98
predicted: scenic actual: scenic
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
Image: /home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2835-2.JPG
Predicted class: document with confidence 1.00
predicted: document actual: document
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
Image: /home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2838.JPG
Predicted class: document with confidence 1.00
predicted: document actual: document
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
Image: /home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2842-2.JPG
Predicted class: document with confidence 0.99
predicted: document actual: document
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
Image: /home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2860.JPG
Predicted class: document with confidence 1.00
predicted: document actual: document
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
Image: /home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2861.JPG
Predicted class: document with confidence 1.00
predicted: document actual: document
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
Image: /home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2864.JPG
Predicted class: document with confidence 1.00
predicted: document actual: document
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
Image: /home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2874-2.JPG
Predicted class: document with confidence 1.00
predicted: document actual: document
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
This image most likely belongs to scenic with a 52.67 percent confidence.

"""

# Define directories
base_path = '/home/madhekar/work/home-media-app/models'
train_dir = os.path.join(base_path, 'img_classify/training')
validation_dir = os.path.join(base_path, 'img_classify/validation')
image_size = (224, 224)
batch_size = 32

# Use ImageDataGenerator to load and augment images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the pre-trained MobileNetV2 model (without the top classification layer)
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model layers to prevent re-training
base_model.trainable = False

# Add a new classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x) # 3 classes: document, scenic, people

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

'''
# 4. Define the EarlyStopping callback with patience
# Monitor 'val_loss' and wait for 5 epochs without improvement before stopping
early_stopping_callback = EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=5,          # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True, # Restore model weights from the epoch with the best value of the monitored metric
    verbose=1            # Print messages when early stopping is triggered
)

# 5. Train the model with early stopping
history = model.fit(
    x_train, y_train,
    epochs=50,  # Set a sufficiently large number of epochs, as early stopping will manage the actual stopping point
    validation_data=(x_val, y_val),
    callbacks=[early_stopping_callback]
)
'''
# Monitor 'val_loss' and wait for 5 epochs without improvement before stopping
early_stopping_callback = EarlyStopping(
    monitor='val_loss',        # Metric to monitor
    patience=5,                # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True, # Restore model weights from the epoch with the best value of the monitored metric
    verbose=1                  # Print messages when early stopping is triggered
)

# Train the model
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    callbacks =[early_stopping_callback]
)

model.save('img_cl.keras')
# Optional: Unfreeze and fine-tune the model for better accuracy
# base_model.trainable = True
# model.compile(...)
# model.fit(...)

# Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)

# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
#plt.savefig("test")
#plt.show()

# Make a prediction on a new image
def predict_image(image_path, model, class_names):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    img_array = img_array / 255.0 # Rescale pixels

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    print(f"Image: {image_path}")
    print(f"Predicted class: {predicted_class} with confidence {confidence:.2f}")
    return predicted_class

# Example usage (assuming you have a test image named 'test_image.jpg')
class_names = list(train_generator.class_indices.keys())
img_list = [
    {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/76b281d8-f830-4f24-9b45-ec02e88b52ac.jpg", "label": "people"},
    {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/8109e957-ce86-4283-96aa-11c55d2fba62-1.jpg", "label": "people"},
    {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/129c39e4-e7bf-4cfd-9bdd-0eb541bf9a60.jpg", "label": "scenic"},
    {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/679e1665-6c43-4018-8a8f-ffb2f33739b6-1.jpg", "label": "scenic"},
    {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/7f5a9544-f625-4bbf-ad25-e0b874a4e5fd-1.jpg", "label": "document"},
    {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/107e99ff-289f-4d92-b2e6-c4943dccbccd.jpg", "label": "document"},
    {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/91f2dc11-cffd-4d08-8575-7d437ef31774.jpg", "label": "scenic"},
    {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/121b15b7-81b3-4792-b654-c95aa2ee4b49-1.jpg", "label": "scenic"},
    {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2835-2.JPG", "label": "document"}, 
    {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2838.JPG", "label": "document"}, 
    {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2842-2.JPG", "label": "document"}, 
    {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2860.JPG", "label": "document"}, 
    {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2861.JPG", "label": "document"}, 
    {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2864.JPG", "label": "document"}, 
    {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2874-2.JPG", "label": "document"}

]
for d in img_list:
  p_class = predict_image(d["img"], model, class_names)
  print(f'predicted: {p_class} actual: {d["label"]}' )



# Assume you have a new image named 'new_image.jpg' in the project directory
new_image_path = "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/6465dc91-6216-4b0e-81ee-3ad87eabe623-4.jpg"

img = load_img(new_image_path, target_size=image_size)
img_array = img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
        class_names[np.argmax(score)], 100 * np.max(score)
    )
)