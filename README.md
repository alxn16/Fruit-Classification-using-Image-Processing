
# Fruit Classification using Image Processing

This project demonstrates fruit classification using deep learning and image processing techniques in TensorFlow, utilizing the VGG16 pre-trained model.

## Project Setup

### 1. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Unzip Dataset
```python
import zipfile
!unzip "/content/drive/MyDrive/fruits.zip"
```

### 3. Data Summary
```python
import os
for dirpath, dirname, filenames in os.walk("train"):
  print(f"There are {len(dirname)} directories and {len(filenames)} images in {dirpath}")
```

## Dependencies
- TensorFlow
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn

Install any missing dependencies using:
```bash
!pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

### 4. Data Augmentation & Preprocessing
```python
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

augment_datagen = ImageDataGenerator(rotation_range=20,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         rescale=1/255.,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1/255.)

train_data = augment_datagen.flow_from_directory(train_dir,
                                                 target_size=(100, 100),
                                                 batch_size=64,
                                                 class_mode="categorical",
                                                 shuffle=True)
test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(100, 100),
                                             batch_size=64,
                                             class_mode="categorical")
```

## Model Training

### 5. Define the VGG16-based Model
```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

VGG16_model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(33, activation='softmax')
])

VGG16_model.summary()

# Compile the model
VGG16_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
```

### 6. Model Training
```python
history_VGG16 = VGG16_model.fit(train_data,
                                epochs=1,
                                steps_per_epoch=len(train_data),
                                validation_data=test_data,
                                validation_steps=len(test_data))
```

## Visualization

### 7. Plot Loss Curves
```python
import matplotlib.pyplot as plt

def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

plot_loss_curves(history_VGG16)
```

## Prediction

### 8. Load and Prepare Image for Prediction
```python
from PIL import Image
import tensorflow as tf

def load_and_prep_image(filename, img_shape=100):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.
    return img
```

### 9. Predict and Plot Results
```python
def pred_and_plot(model, filename, class_names):
    img = load_and_prep_image(filename)
    pred = model.predict(tf.expand_dims(img, axis=0))

    if len(pred[0]) > 1:
        pred_class = class_names[pred.argmax()]
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]

    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)

pred_and_plot(VGG16_model, "/content/Fruits/test/Cherry/Cherry_100.jpg", class_names)
```

## Conclusion

This project demonstrates how transfer learning using the VGG16 model can be applied to a custom fruit classification task. With a dataset of fruit images, we trained a model that predicts the type of fruit in a given image. This serves as a basic introduction to transfer learning and image classification in TensorFlow.

