import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import os

# Define image properties
img_size = (64, 64)  # Adjustable based on dataset
batch_size = 32
epochs = 25
dataset_path = r"D:\Practice Code\static\AtoZ"  # Update with your dataset path

# Image Data Generator (Preprocessing)
datagen = ImageDataGenerator(
    rescale=1./255,  
    rotation_range=10,  
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    shear_range=0.1,  
    zoom_range=0.1,  
    validation_split=0.2  
)

# Load training images (Grayscale mode)
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",  # Ensuring grayscale (1 channel)
    class_mode="categorical",
    subset="training"
)

# Load validation images
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation"
)

# Number of classes (A-Z or A-Z + numbers)
num_classes = len(train_data.class_indices)

# Optimized CNN Model for Skeletonized Hand Signs
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    BatchNormalization(),  
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_data, validation_data=val_data, epochs=epochs)

# Save Model
model.save("isl_hand_sign_model.h5")
print("Model saved successfully!")
