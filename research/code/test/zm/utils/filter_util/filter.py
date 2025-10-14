import os
import ast
import numpy as np
import json
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
from utils.config_util import config

    # Define directories, files, params
def load_init_params():
    (
        filter_model_path,
        train_data_path,
        validation_data_path,
        traing_data_path,
        trainging_map_file,
        filter_model_name,
        filter_model_classes,
        image_size,
        batch_size,
    ) = config.filer_config_load()
    return filter_model_path, train_data_path, validation_data_path, traing_data_path, trainging_map_file, filter_model_name, filter_model_classes, image_size, batch_size

def train_filter_model(train_dir, validation_dir, model_file, classes_file, image_size, batch_size):

    print(f'{train_dir}:{validation_dir}:{model_file}:{classes_file}:{image_size}:{batch_size}')
    batch_size_int = int(batch_size)
    sz = ast.literal_eval(image_size)
    x, y = sz[0], sz[1]
    print(x, y)
    image_size_int = (int(x), int(y))

    # Use ImageDataGenerator to load and augment images
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size_int,
        batch_size=batch_size_int,
        class_mode="categorical",
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=image_size_int,
        batch_size=batch_size_int,
        class_mode="categorical",
    )

    # Load the pre-trained MobileNetV2 model (without the top classification layer)
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

    # Freeze the base model layers to prevent re-training
    base_model.trainable = False

    # Add a new classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    predictions = Dense(3, activation="softmax")(
        x
    )  # 3 classes: document, scenic, people

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(model)
    # Monitor 'val_loss' and wait for 5 epochs without improvement before stopping
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",  # Metric to monitor
        patience=5,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored metric
        verbose=1,  # Print messages when early stopping is triggered
    )
    """
    /home/madhekar/work/home-media-app/models/image_classify_filter/training
    /home/madhekar/work/home-media-app/models/image_classify_filter/training
    /home/madhekar/work/home-media-app/models/image_classify_filter/validation
    """
    # Train the model
    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        callbacks=[early_stopping_callback],
    )

    model.save(model_file)

    joblib.dump(train_generator.class_indices, classes_file)
    #np.save( classes_file, train_generator.class_indices)
    #return list(train_generator.class_indices.keys())

# Make a prediction on a new image
def predict_image(image_path, model, class_names, image_size):
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
#class_names = list(train_generator.class_indices.keys())
def test_model(model, class_names, testing_path, Testing_map_file):
    with open(os.path.join(testing_path, Testing_map_file)) as f:
        dlist = json.load(f)
        print('--->', dlist)
        for d in dlist:
            p_class = predict_image(d["img"], model, class_names)
            print(f'predicted: {p_class} actual: {d["label"]}' )  
    # img_list = [
    #     {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/76b281d8-f830-4f24-9b45-ec02e88b52ac.jpg", "label": "people"},
    #     {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/8109e957-ce86-4283-96aa-11c55d2fba62-1.jpg", "label": "people"},
    #     {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/129c39e4-e7bf-4cfd-9bdd-0eb541bf9a60.jpg", "label": "scenic"},
    #     {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/679e1665-6c43-4018-8a8f-ffb2f33739b6-1.jpg", "label": "scenic"},
    #     {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/7f5a9544-f625-4bbf-ad25-e0b874a4e5fd-1.jpg", "label": "document"},
    #     {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/107e99ff-289f-4d92-b2e6-c4943dccbccd.jpg", "label": "document"},
    #     {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/91f2dc11-cffd-4d08-8575-7d437ef31774.jpg", "label": "scenic"},
    #     {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/121b15b7-81b3-4792-b654-c95aa2ee4b49-1.jpg", "label": "scenic"},
    #     {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2835-2.JPG", "label": "document"}, 
    #     {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2838.JPG", "label": "document"}, 
    #     {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2842-2.JPG", "label": "document"}, 
    #     {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2860.JPG", "label": "document"}, 
    #     {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2861.JPG", "label": "document"}, 
    #     {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2864.JPG", "label": "document"}, 
    #     {"img": "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/IMG_2874-2.JPG", "label": "document"}

    # ]
  


def execute():

    fmp, trdp, vdp, tsdp, tmf, fmn, fmc,isz, bsz = load_init_params()

    train_filter_model(trdp, vdp, os.path.join(fmp, fmn), os.path.join(fmp, fmc), isz, bsz)

    model = load_model(os.path.join(fmp, fmn))

    class_names = joblib.load(os.path.join(fmp, fmc))

    print(class_names)
    
    test_model(model, class_names, tsdp, tmf)
   
  