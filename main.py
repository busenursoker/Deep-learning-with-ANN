import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet import preprocess_input 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential, Model  
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split


label = []
path = []
fish_dir = '/kaggle/input/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset' 

for dirname, _, filenames in os.walk(fish_dir):
    for filename in filenames:
        if os.path.splitext(filename)[-1] == '.png':
            if dirname.split()[-1] != 'GT':  
                label.append(os.path.split(dirname)[-1])
                path.append(os.path.join(dirname, filename))

# Create DataFrame
data = pd.DataFrame(columns=['path', 'label'])
data['path'] = path
data['label'] = label

# Display the tail of the data
print(data.tail(),"\n\n")
data.info()

idx = 0
plt.figure(figsize=(5,5))
for unique_label in data['label'].unique():
    plt.subplot(3, 3, idx+1)
    plt.imshow(plt.imread(data[data['label']==unique_label].iloc[0,0]))
    plt.title(unique_label)
    plt.axis('off')
    idx+=1 

plt.figure(figsize=(5,2))
sns.countplot(y='label', data=data) 
plt.title('Label Count')
plt.show()

train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)
print(train_data.shape)
print(test_data.shape)

# Constants
img_size = 128  # Resize the images to a fixed size
batch_size = 32

# Preprocessing function
def preprocess_image(img_path, img_size):
    img = load_img(img_path, target_size=(img_size, img_size))
    img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
    img_array = img_array.flatten()  # Flatten the image (necessary for ANN)
    return img_array

# Apply preprocessing to the dataset
train_images = np.array([preprocess_image(p, img_size) for p in train_data['path']])
test_images = np.array([preprocess_image(p, img_size) for p in test_data['path']])

# Encode labels to integers
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_data['label'])
test_labels = label_encoder.transform(test_data['label'])

# Convert labels to categorical (one-hot encoding)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Define the ANN model
model = Sequential()

# Input layer
model.add(Dense(512, input_shape=(img_size * img_size * 3,), activation='relu'))  # First hidden layer
model.add(Dense(256, activation='relu'))  # Second hidden layer
model.add(Dense(128, activation='relu'))  # Third hidden layer

# Output layer (number of unique classes)
num_classes = len(data['label'].unique())
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model architecture
model.summary()

# Train the model
history = model.fit(train_images, train_labels, 
                    validation_data=(test_images, test_labels),
                    epochs=30, batch_size=batch_size, verbose=1)

# Plot training history
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Loss')
plt.legend()

plt.show()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc:.4f}")

# Make predictions
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

print("\n\nClassification Report:\n\n", classification_report(true_classes, predicted_classes, target_names=label_encoder.classes_))
cm = confusion_matrix(true_classes, predicted_classes)

# Confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()


# Define a function to create the ANN model
def create_model(units=32, activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(units, input_shape=(img_size * img_size * 3,), activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameter search space
units_options = [32, 64, 128]
activation_options = ['relu', 'tanh']
optimizer_options = ['adam', 'sgd']
batch_size_options = [32, 64]
epochs_options = [10, 20]

# Manual grid search
best_accuracy = 0
best_params = None

for units in units_options:
    for activation in activation_options:
        for optimizer in optimizer_options:
            for batch_size in batch_size_options:
                for epochs in epochs_options:
                    print(f"Training with units={units}, activation={activation}, optimizer={optimizer}, batch_size={batch_size}, epochs={epochs}")
                    model = create_model(units=units, activation=activation, optimizer=optimizer)
                    
                    # Train the model
                    history = model.fit(train_images, train_labels, 
                                        validation_data=(test_images, test_labels),
                                        epochs=epochs, batch_size=batch_size, verbose=0)
                    
                    # Evaluate the model
                    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
                    print(f"Test Accuracy: {test_acc:.4f}")
                    
                    # Track the best performing hyperparameters
                    if test_acc > best_accuracy:
                        best_accuracy = test_acc
                        best_params = {'units': units, 'activation': activation, 'optimizer': optimizer, 
                                       'batch_size': batch_size, 'epochs': epochs}

print(f"Best accuracy: {best_accuracy:.4f}")
print(f"Best hyperparameters: {best_params}")
