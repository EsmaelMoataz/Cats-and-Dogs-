import warnings
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model, img_to_array, load_img
from IPython.display import Image as IPImage, display
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import pandas as pd
from PIL import UnidentifiedImageError
from tqdm import tqdm

# Constants
TRAIN_DIR = '/Users/esmaelmoataz/Documents/Machine learning/Data Model/train'
VALIDATION_DIR = '/Users/esmaelmoataz/Documents/Machine learning/Data Model/validation'
TEST_DIR = '/Users/esmaelmoataz/Documents/Machine learning/Data Model/test'
MODEL_SAVE_PATH = '/Users/esmaelmoataz/Documents/Machine learning/Data Model/cat_dog_others_detector.h5'
MODEL_PLOT_PATH = '/Users/esmaelmoataz/Documents/Machine learning/Data Model/model_plot.png'
CLASSIFIED_TEST_DIR = '/Users/esmaelmoataz/Documents/Machine learning/Data Model/classified_test'
TARGET_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 140
LEARNING_RATE = 0.001  # Adjusted learning rate to a more typical value

# Helper functions
def create_directory(directory):
    """Create directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def list_files_in_directory(directory):
    """List the number of files in a directory."""
    file_count = sum(len(files) for _, _, files in os.walk(directory))
    print(f"Directory {directory} contains {file_count} files.")

def check_and_preprocess_images(directory):
    """Check and preprocess images in a directory."""
    total_files, removed_files = 0, 0
    for root, _, files in os.walk(directory):
        total_files += len(files)
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                file_path = os.path.join(root, file)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", UserWarning)
                        img = load_img(file_path)
                        if img.mode in ['P', 'RGBA']:
                            img = img.convert('RGBA')
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(file_path)
                except (UnidentifiedImageError, OSError, UserWarning):
                    print(f"Removing corrupted image: {file_path}")
                    os.remove(file_path)
                    removed_files += 1
    print(f"Processed {total_files} files in {directory}, removed {removed_files} corrupted files.")

def load_and_preprocess_image(img_path, target_size=TARGET_SIZE):
    """Load and preprocess a single image."""
    img = load_img(img_path, target_size=target_size)
    if img.mode in ['P', 'RGBA']:
        img = img.convert('RGBA')
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def classify_single_image(img_path):
    """Classify a single image and display the result."""
    img_array = load_and_preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    predicted_label = class_labels[predicted_class]
    print(f"The image is classified as: {predicted_label}")
    img = load_img(img_path)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()

# Create required directories
for directory in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR, CLASSIFIED_TEST_DIR]:
    create_directory(directory)

# List files in directories
for directory in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:
    print(f"{directory} contents:")
    list_files_in_directory(directory)

# Check and preprocess images
for directory in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:
    check_and_preprocess_images(directory)

# Data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

# Load and compile the model
model = tf.keras.models.load_model(MODEL_SAVE_PATH)
model.summary()
plot_model(model, to_file=MODEL_PLOT_PATH, show_shapes=True, show_layer_names=True)
display(IPImage(filename=MODEL_PLOT_PATH))

# Define a custom callback for real-time plotting
class RealTimePlotting(tf.keras.callbacks.Callback):
    def __init__(self):
        super(RealTimePlotting, self).__init__()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        plt.ion()
        self.ax1.set_title('Model accuracy')
        self.ax1.set_ylabel('Accuracy')
        self.ax1.set_xlabel('Epoch')
        self.ax2.set_title('Model loss')
        self.ax2.set_ylabel('Loss')
        self.ax2.set_xlabel('Epoch')
    
    def on_train_begin(self, logs=None):
        self.accuracy = []
        self.val_accuracy = []
        self.loss = []
        self.val_loss = []
    
    def on_epoch_end(self, epoch, logs=None):
        self.accuracy.append(logs['accuracy'])
        self.val_accuracy.append(logs['val_accuracy'])
        self.loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])

        self.ax1.plot(self.accuracy, label='Train Accuracy')
        self.ax1.plot(self.val_accuracy, label='Val Accuracy')
        self.ax2.plot(self.loss, label='Train Loss')
        self.ax2.plot(self.val_loss, label='Val Loss')

        self.ax1.legend()
        self.ax2.legend()
        plt.draw()
        plt.pause(0.001)

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with early stopping and real-time plotting
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=EPOCHS,
    callbacks=[RealTimePlotting()]
)

# Save the model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# Classify and move test images
test_image_files = []
for category in ['cat', 'dog', 'others']:
    category_dir = os.path.join(TEST_DIR, category)
    test_image_files += [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

predictions = [(img_path, np.argmax(model.predict(load_and_preprocess_image(img_path))[0])) for img_path in test_image_files]
class_labels = list(train_generator.class_indices.keys())
predictions_with_labels = [(img_path, class_labels[predicted_class]) for img_path, predicted_class in predictions]

for img_path, label in predictions_with_labels:
    destination_path = os.path.join(CLASSIFIED_TEST_DIR, label, os.path.basename(img_path))
    shutil.move(img_path, destination_path)
    print(f'Image: {os.path.basename(img_path)} is classified as: {label} and moved to {destination_path}')

# Visualize training history after training completes
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Create a DataFrame for predictions
df_predictions = pd.DataFrame(predictions_with_labels, columns=['Image Path', 'Predicted Label'])
display(df_predictions)

# Visualize misclassified images in validation set
validation_generator.reset()
val_predictions = model.predict(validation_generator, steps=validation_generator.samples // validation_generator.batch_size + 1)
val_predicted_classes = np.argmax(val_predictions, axis=1)
true_classes = validation_generator.classes

df_val = pd.DataFrame({
    'Filename': validation_generator.filenames,
    'True Label': [class_labels[k] for k in true_classes],
    'Predicted Label': [class_labels[k] for k in val_predicted_classes]
})

misclassified = df_val[df_val['True Label'] != df_val['Predicted Label']]
print(f"Number of misclassified images: {len(misclassified)}")

fig, axes = plt.subplots(1, min(len(misclassified), 5), figsize=(20, 4))
for i, (idx, row) in enumerate(misclassified.sample(min(len(misclassified), 5)).iterrows()):
    img_path = os.path.join(VALIDATION_DIR, row['Filename'])
    img = load_img(img_path)
    axes[i].imshow(img)
    axes[i].set_title(f"True: {row['True Label']}\nPredicted: {row['Predicted Label']}")
    axes[i].axis('off')
plt.show()

# Example usage of classify_single_image function
# classify_single_image('path_to_image')
