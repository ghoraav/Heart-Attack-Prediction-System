import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

# Custom data generator for loading images
class ECGImageGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, batch_size, img_size, shuffle=True):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.image_paths = []
        self.labels = []
        self._load_data()
        self.indexes = np.arange(len(self.image_paths))
    
    def _load_data(self):
        # Load image paths and labels from the directory
        for class_idx, class_name in enumerate(os.listdir(self.image_dir)):
            class_folder = os.path.join(self.image_dir, class_name)
            if os.path.isdir(class_folder):
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    if img_name.endswith(('png', 'jpg', 'jpeg')):
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)  # 0 for normal, 1 for abnormal

        # Convert labels to numpy array
        self.labels = np.array(self.labels)

    def __len__(self):
        # Number of batches per epoch
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        # Shuffle indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # Generate one batch of data
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.image_paths[i] for i in batch_indexes]
        batch_labels = self.labels[batch_indexes]

        images = np.zeros((self.batch_size, *self.img_size, 3))  # 3 for RGB images
        for i, img_path in enumerate(batch_paths):
            img = load_img(img_path, target_size=self.img_size)
            img = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
            images[i] = img

        return images, batch_labels  # Return images and corresponding labels

# Define parameters
img_width, img_height = 128, 128
batch_size = 32

# Define the paths
image_dir = r'C:\Users\ghora\Desktop\AI-PJ\dataset'

# Create the train and validation generators
train_generator = ECGImageGenerator(image_dir=image_dir, batch_size=batch_size, img_size=(img_width, img_height))
validation_generator = ECGImageGenerator(image_dir=image_dir, batch_size=batch_size, img_size=(img_width, img_height))

# Model Definition (same as before)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    class_weight={0: 1, 1: 1},  # Example class weights, adjust as needed
)

# Save the final model
model.save('ecg_model.keras')
