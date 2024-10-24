# Fish Species Classification Using Artificial Neural Networks (ANN)

## Project Overview

This project aims to classify fish species from images using a custom-built Artificial Neural Network (ANN). We worked with a large-scale fish dataset from Kaggle, containing multiple species. The model processes image data, trains an ANN to recognize the species based on patterns in the images, and achieves a notable accuracy after thorough preprocessing and hyperparameter tuning.

- The Data set i've used in this project : https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset
- Kaggle link to see the output and data visuals : https://www.kaggle.com/code/busenursker/deep-learning

---

## Project Workflow

### 1. Data Collection

- The dataset consists of fish species images organized by directories based on species names.
- The data is stored in the `/Fish_Dataset/` folder and each subdirectory contains images of a specific fish species.
  
### 2. Data Preprocessing

To prepare the data for model training, several preprocessing steps were applied:
- **Image Resizing**: Each image is resized to a uniform size of `128x128` pixels to ensure consistency across all input data.
- **Normalization**: Pixel values are scaled to fall between `[0, 1]`, which improves the stability and efficiency of the model during training.
- **Label Encoding**: The species names (labels) are converted into integer values using `LabelEncoder`. These integers are further one-hot encoded for use in multi-class classification.

```python
img_size = 128  # Resize the images to a fixed size
img = load_img(img_path, target_size=(img_size, img_size))
img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
```
### 3. Model Architecture

The model is a basic Artificial Neural Network (ANN), which consists of three fully connected layers:
- **Input Layer**: Input shape is flattened (`128 * 128 * 3`), since we are feeding 128x128 images with 3 color channels (RGB).
- **Hidden Layers**: The ANN has three dense layers with 512, 256, and 128 neurons respectively, each followed by a `ReLU` activation function to learn non-linear patterns.
- **Output Layer**: The output layer uses a `softmax` activation to predict the probability distribution across multiple fish species (classes).

```python
model.add(Dense(512, input_shape=(img_size * img_size * 3,), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  # Output layer
```
### 4. Model Compilation and Training

- **Optimizer**: `Adam` optimizer was used for its adaptive learning rate, helping the model converge faster.
- **Loss Function**: `categorical_crossentropy` is employed since we are dealing with multi-class classification.
- **Metrics**: Accuracy is the primary metric used to evaluate model performance.

Training was done for 30 epochs with a batch size of 32. Early Stopping was applied to prevent overfitting.

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, 
                    validation_data=(test_images, test_labels),
                    epochs=30, batch_size=batch_size)
```

### 5. Evaluation and Results

The model is evaluated on the test set, achieving a test accuracy between 80-85%.

- **Test Accuracy**: Represents model performance on unseen data.
- **Loss**: A lower loss indicates better performance.

```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
```

Other metrics include:

- A classification report showing precision, recall, and F1-score for each fish species.
- A confusion matrix illustrating where the model struggles to distinguish between species.

### 6. Hyperparameter Tuning

To optimize performance, the following hyperparameters were tuned:

Number of neurons in hidden layers `(e.g., 32, 64, 128)` to determine optimal learning capacity.
- **Activation Functions**: Compared `ReLU` and `tanh` to find which yielded better accuracy.
- **Optimizers**: `Adam `proved better than `SGD`.
- **Batch Size and Epochs**: Experimented with batch sizes (32 and 64) and trained for various epochs (10, 20, 30).

This tuning helped achieve the model's final accuracy.

### Lessons Learned

**Achieving High Accuracy**
- Data Preprocessing: Consistent image resizing and normalization were crucial for model stability.
- Hyperparameter Tuning: Experimenting with different values for neurons, activation functions, and optimizers improved accuracy.
- Avoiding Overfitting: Early stopping was key to prevent overfitting by monitoring validation loss.

**Challenges**
- Training Time: Large image datasets can take time to train. Using a GPU or simplifying the model may speed up training.
- Generalization: The model generalizes well but using data augmentation or transfer learning might enhance performance on unseen data.



