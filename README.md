# Sign Language Alphabet Recognition Using CNN and MobileNetV2

A deep learning project that recognizes static hand gestures representing English alphabet letters using the **Sign Language MNIST dataset**. This project implements a custom Convolutional Neural Network (CNN) from scratch and compares it with a pretrained MobileNetV2 model using transfer learning and fine-tuning.

The project covers the complete deep learning workflow, including dataset exploration, image preprocessing, model training, performance evaluation, confusion matrix visualization, model saving, and prediction on uploaded images.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [Models Implemented](#models-implemented)
- [Image Preprocessing](#image-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Installation and Setup](#installation-and-setup)
- [How to Run](#how-to-run)
- [Making Predictions](#making-predictions)
- [Saved Models](#saved-models)
- [Results and Performance Analysis](#results-and-performance-analysis)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Learning Outcomes](#learning-outcomes)
- [References](#references)

## Project Overview

Sign language is an important means of communication for people with hearing or speech impairments. Computer vision and deep learning can help recognize hand gestures and make sign recognition more accessible.

This project builds an image classification system that identifies static hand gestures corresponding to English alphabet letters. It uses the Sign Language MNIST dataset, which contains grayscale images of hand gestures.

Two deep learning approaches are implemented:

1. **Custom CNN:** A convolutional neural network trained from scratch to learn gesture-specific visual features.
2. **MobileNetV2:** A pretrained convolutional neural network adapted to the task using transfer learning, followed by fine-tuning of selected backbone layers.

The models are evaluated using test accuracy, test loss, classification reports, and confusion matrices. The trained models can also be used to predict a letter from an uploaded image.

**Project type:** Computer Vision | Deep Learning | Image Classification

## Objectives

- Build a sign language alphabet classifier using deep learning.
- Understand and preprocess the Sign Language MNIST dataset.
- Implement a custom CNN using TensorFlow and Keras.
- Apply transfer learning using pretrained MobileNetV2 weights.
- Fine-tune selected MobileNetV2 layers for the target classification task.
- Evaluate models using accuracy, loss, precision, recall, and F1-score.
- Visualize training performance and class-wise prediction errors.
- Save trained models for reuse without retraining.
- Test model predictions on uploaded images.

## Dataset

This project uses the **Sign Language MNIST** dataset.

Dataset source: [Sign Language MNIST on Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

The dataset contains grayscale images representing hand gestures for 24 English alphabet letters. The letters **J and Z are excluded** because they require motion and cannot be fully represented by a single static image.

### Dataset characteristics

| Property | Description |
|---|---|
| Dataset | Sign Language MNIST |
| Task | Multiclass image classification |
| Input image size | 28 × 28 pixels |
| Original image channels | 1 (grayscale) |
| Number of classes | 24 |
| Excluded letters | J and Z |
| Label format | Integer class labels |
| Dataset format | CSV files containing labels and pixel values |

The notebook downloads the dataset using `kagglehub`, loads the training and testing CSV files, separates the labels from the pixel values, and reshapes the image data into 28 × 28 arrays.

The training dataset is further divided into training and validation subsets using a stratified split, with 15% of the original training data reserved for validation.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow | Deep learning framework |
| Keras | Model construction, training, and inference |
| MobileNetV2 | Pretrained feature extraction and transfer learning |
| NumPy | Numerical operations and array manipulation |
| Pandas | Dataset loading and inspection |
| Matplotlib | Image display and training plots |
| Seaborn | Confusion matrix visualization |
| Scikit-learn | Data splitting and classification metrics |
| OpenCV | Image resizing |
| Pillow (PIL) | Image loading and conversion |
| KaggleHub | Dataset download |

The notebook is designed for execution in Google Colab or a compatible Python environment with the required dependencies installed.

## Project Workflow

The project follows these stages:

1. **Environment setup:** Import libraries and initialize random seeds.
2. **Dataset acquisition:** Download and load Sign Language MNIST.
3. **Data exploration:** Inspect labels, pixel values, image shapes, and sample gestures.
4. **Image preprocessing:** Normalize pixel values and add the channel dimension.
5. **Label encoding:** Map the original labels to 24 consecutive class indices.
6. **Data splitting:** Create stratified training and validation sets.
7. **Custom CNN training:** Train a CNN from scratch with early stopping and checkpointing.
8. **Transfer learning:** Use MobileNetV2 pretrained on ImageNet with a custom classification head.
9. **Fine-tuning:** Unfreeze the final 30 backbone layers and keep batch normalization layers frozen.
10. **Model evaluation:** Compare test performance and examine class-wise predictions.
11. **Model export:** Save the trained models in Keras format.
12. **Image prediction:** Upload a new image and obtain a predicted alphabet letter with a confidence score.

## Models Implemented

### 1. Custom Convolutional Neural Network (CNN)

The first model is built from scratch using Keras Sequential layers.

Its architecture uses convolutional layers to extract visual patterns from hand gestures, max-pooling layers to reduce spatial dimensions, dropout to help reduce overfitting, and dense layers for classification.

**Input shape:** `28 × 28 × 1`

**Output classes:** 24

The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss.

Early stopping monitors validation loss, while model checkpointing saves the best-performing checkpoint.

This model provides a baseline for understanding how a CNN learns features directly from the sign language images.

### 2. MobileNetV2 with Transfer Learning

The second approach uses MobileNetV2 pretrained on ImageNet.

Since MobileNetV2 expects three-channel images, the grayscale input images are converted to RGB and resized to 96 × 96 pixels.

The pretrained backbone is initially frozen so that the newly added classification head can learn the target classes without immediately modifying the pretrained weights.

The classification head consists of:

- A frozen MobileNetV2 feature extractor.
- `GlobalAveragePooling2D` to aggregate spatial features.
- A dropout layer with a rate of 0.3.
- A dense layer containing 128 neurons with ReLU activation.
- A second dropout layer with a rate of 0.2.
- A final dense layer containing 24 neurons with softmax activation.

The model uses the Adam optimizer with an initial learning rate of `0.001` and sparse categorical cross-entropy loss.

### 3. MobileNetV2 Fine-Tuning

After the initial transfer learning stage, the model is fine-tuned to adapt its higher-level visual features to sign language gestures.

The fine-tuning stage:

- Unfreezes the MobileNetV2 backbone.
- Keeps the earlier backbone layers frozen.
- Makes the final 30 backbone layers trainable.
- Keeps batch normalization layers frozen for stability.
- Reduces the learning rate to `0.0001`.
- Uses early stopping to monitor validation loss.

This approach allows selected pretrained weights to adapt to the target dataset while retaining the general visual representations learned during pretraining.

## Image Preprocessing

The project uses separate preprocessing pipelines for the two models.

### Preprocessing for the custom CNN

1. Read the image data from CSV files.
2. Reshape each sample into a 28 × 28 grayscale image.
3. Convert pixel values to `float32`.
4. Normalize pixel values from `[0, 255]` to `[0, 1]`.
5. Add a single-channel dimension, producing an input shape of `(28, 28, 1)`.
6. Encode the original labels into consecutive class indices from 0 to 23.

### Preprocessing for MobileNetV2

1. Start with the normalized grayscale images.
2. Convert grayscale images into three-channel RGB images.
3. Resize images to 96 × 96 pixels.
4. Scale pixel values from `[0, 1]` to `[-1, 1]`.
5. Construct TensorFlow datasets using batching, shuffling for training, parallel mapping, and prefetching.

The MobileNetV2 preprocessing pipeline is designed to match the input requirements of the pretrained model.

## Model Training and Evaluation

Both approaches are evaluated using the held-out test dataset.

### Evaluation metrics

**Test accuracy:** Measures the proportion of correctly classified test samples.

**Test loss:** Measures the classification loss on the test dataset.

**Precision:** Measures how often predictions for a class are correct.

**Recall:** Measures how many actual samples of a class are correctly identified.

**F1-score:** Combines precision and recall into a single metric.

**Confusion matrix:** Shows the relationship between actual and predicted classes, helping identify frequently confused gestures.

The notebook also plots training and validation accuracy and loss curves for the custom CNN. Classification reports and confusion matrices are generated for the CNN and MobileNetV2 approaches, including the fine-tuned model.

### Performance comparison

The following table can be completed after running the notebook and recording the actual test results.

| Metric | Custom CNN | MobileNetV2 (initial) | MobileNetV2 (fine-tuned) |
|---|---|---|---|
| Test accuracy | Add measured result | Add measured result | Add measured result |
| Test loss | Add measured result | Add measured result | Add measured result |
| Training approach | From scratch | Transfer learning | Fine-tuning |

No numerical performance results are stated here because the actual execution results must be measured and verified.


## Installation and Setup

### Prerequisites

- Python 3.10 or another version compatible with your selected TensorFlow release.
- pip.
- Jupyter Notebook or Google Colab.
- A GPU is recommended for faster training but is not mandatory.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/sign-language-recognition-cnn-mobilenetv2.git

cd sign-language-recognition-cnn-mobilenetv2
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

Create a `requirements.txt` file containing the following packages:

```text
tensorflow
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
Pillow
kagglehub
notebook
```

Install the packages:

```bash
pip install -r requirements.txt
```

For a reproducible setup, use package versions tested with your actual notebook and hardware rather than assuming that every latest release is compatible.

## How to Run

### Option 1: Google Colab

Google Colab is a convenient option because the notebook uses the Colab file-upload interface for external image predictions.

1. Open the notebook from the repository in Google Colab.
2. Select a runtime with sufficient memory; GPU acceleration is recommended for training.
3. Run the notebook cells in order.
4. Allow KaggleHub to download the dataset.
5. Inspect the dataset and run the preprocessing cells.
6. Train and evaluate the custom CNN.
7. Train the initial MobileNetV2 model.
8. Fine-tune MobileNetV2 and evaluate it again.
9. Run the image-upload prediction cells to test the saved models.

Dataset downloads may require internet access. Follow any Kaggle dataset access requirements if prompted.

### Option 2: Local Jupyter Notebook

After installing the dependencies, launch Jupyter:

```bash
jupyter notebook
```

Open `Sing_laguage_CNN_Mobilenet.ipynb` and execute the cells sequentially.

The notebook's image-upload functions import `files` from `google.colab`. Those functions are specific to Google Colab. To use them in local Jupyter, replace the upload mechanism with a local image path or a suitable notebook file-upload widget.

## Making Predictions

The notebook includes inference functions for both saved models.

### Custom CNN prediction

The custom CNN expects a grayscale image resized to 28 × 28 pixels.

The prediction pipeline:

1. Loads an uploaded image.
2. Converts it to grayscale.
3. Resizes it to 28 × 28.
4. Normalizes its pixel values to `[0, 1]`.
5. Adds the channel and batch dimensions.
6. Runs the model to obtain class probabilities.
7. Selects the highest-probability class and displays the predicted letter and confidence score.

### MobileNetV2 prediction

The MobileNetV2 model expects a three-channel image resized to 96 × 96 pixels.

The prediction pipeline:

1. Loads an uploaded image and converts it to RGB.
2. Resizes it to 96 × 96.
3. Normalizes its pixel values to `[-1, 1]`.
4. Adds a batch dimension.
5. Runs the model to obtain class probabilities.
6. Displays the predicted alphabet letter and confidence score.

The confidence score is the model's predicted probability for the selected class. It should not automatically be interpreted as a guarantee that the prediction is correct.

**Note:** Images from external sources may differ from the training dataset in background, lighting, hand orientation, cropping, and contrast. Real-world prediction performance may therefore differ from test-set performance.

## Saved Models

The notebook saves trained models using the Keras `.keras` format.

| Model file | Purpose |
|---|---|
| `model_sing_language_CNN.keras` | Saved custom CNN model |
| `model_sing_language_mobilenetv2.keras` | Saved MobileNetV2 model after training and fine-tuning |

The notebook also uses checkpoint files during training:

- `sign_language_CNN.keras` — best custom CNN checkpoint.
- `mobilenetv2_best.keras` — best checkpoint from the initial MobileNetV2 training stage.

The checkpoint and final-export filenames are separate. Run the relevant saving cells to create the corresponding files.

Keras model files can be large, so consider GitHub's file-size limits before committing them. If the files are too large, link to an appropriate model-hosting service instead.

## Results and Performance Analysis

The project evaluates both architectures and generates visual outputs that can be used to analyze their behavior.

The main outputs include:

- Sample images from the dataset.
- Training and validation accuracy curves.
- Training and validation loss curves for the custom CNN.
- Test accuracy and loss.
- Classification reports for each model.
- Confusion matrices for the 24 classes.
- Predicted letters and confidence scores for uploaded images.

Use these outputs to compare the models based on measured performance, class-wise errors, training time, and computational requirements.

To make the GitHub repository more informative, export the plots from your notebook and add them to a `results/` directory. Insert the actual results into this section after executing the notebook.

## Limitations

- The system recognizes static alphabet gestures rather than continuous sign language.
- The letters J and Z are not included because their gestures require movement.
- The dataset contains small grayscale images, which differ from many real-world camera images.
- Predictions on external images can be affected by lighting, backgrounds, hand positioning, and image quality.
- A high model confidence does not necessarily mean a prediction is correct.
- The current notebook provides image classification, not real-time webcam recognition, sentence translation, or full sign language interpretation.

## Future Improvements

Possible extensions include:

- Build a web application using Streamlit or Flask.
- Add real-time webcam-based alphabet recognition.
- Display top-k predictions and more informative confidence visualizations.
- Test performance on images captured in varied real-world conditions.
- Add data augmentation to improve robustness.
- Experiment with alternative architectures and hyperparameters.
- Compare inference speed, model size, and class-wise performance.
- Extend the project toward dynamic gesture recognition using video sequences.
- Explore sentence-level sign language recognition as a separate, more advanced task.

## Learning Outcomes

Through this project, the following concepts can be explored:

- Image preprocessing and normalization.
- Multiclass classification.
- CNN architecture design.
- Transfer learning with pretrained neural networks.
- Fine-tuning deep learning models.
- Dropout and early stopping.
- Model checkpointing and saving.
- Classification metrics and confusion matrix analysis.
- Prediction using trained Keras models.
- Reproducible machine learning experimentation.

## References

1. Sign Language MNIST dataset: https://www.kaggle.com/datasets/datamunge/sign-language-mnist
2. TensorFlow documentation: https://www.tensorflow.org/
3. Keras documentation: https://keras.io/
4. MobileNetV2 documentation: https://keras.io/api/applications/mobilenet/
5. Scikit-learn documentation: https://scikit-learn.org/stable/

## Author

MO ALFAIZ MANSOORI

- GitHub: https://github.com/MOALFAIZMANSOORI
- LinkedIn: https://www.linkedin.com/in/alfaizmansoori/



---

If you find this project useful, consider giving the repository a star ⭐.
