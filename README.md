# ImageClassifier

A bachelor's senior-year milestone project exploring **image classification with deep learning**, developed in two iterations: an initial convolutional neural network (CNN), followed by a VGG-inspired CNN to improve classification performance.

Both versions classify images into the same **10 CIFAR-10 categories** and include a simple **Tkinter desktop interface** for testing the trained model with images selected from a computer. The repository preserves both attempts to show how the approach evolved, from the first working classifier to a more carefully regularized architecture.

## Project at a glance

- **Task:** Multiclass image classification on CIFAR-10.
- **Dataset:** 60,000 color images at 32 × 32 pixels (50,000 training images and 10,000 standard test images).
- **Frameworks:** Python, TensorFlow/Keras, NumPy, OpenCV, scikit-learn, Matplotlib, Seaborn, Pillow, and Tkinter.
- **Deliverables:** Two training notebooks, two saved Keras models, and two desktop GUIs.
- **Reported accuracy:** **73.00%** for the initial CNN and **88.57%** for the VGG-inspired version, using the evaluation setups recorded in their respective notebooks (see [Evaluation note](#evaluation-note)).

## How the project evolved

### Attempt 1 — Custom CNN

**Files:** `CNN(2).ipynb` · `finalimageclassifier(2).keras` · `GUI1.py`

The first version builds a CNN from scratch with a sequence of convolution and pooling layers, followed by fully connected layers for classification.

**Architecture and training:**

- Five convolutional layers with **32 → 64 → 128 → 256 → 512** filters, each followed by max pooling.
- Dense classifier with **512 → 256 → 128 → 64 → 32 → 10** units and a softmax output.
- **Adam** optimizer and **sparse categorical cross-entropy** loss.
- **20 epochs**, batch size **32**.
- Pixel values scaled to **[0, 1]**.
- The original CIFAR-10 10,000-image test split is divided equally: **5,000 for validation** and **5,000 for final evaluation**.

**Recorded result:** **73.00% accuracy** on its 5,000-image evaluation split.

`GUI1.py` provides a basic desktop interface: choose an image using **Open Image**, and the application displays its predicted class.

### Attempt 2 — VGG-Inspired CNN

**Files:** `VGG(2).ipynb` · `finalmodel(2).keras` · `GUI2.py`

The second version was developed to address the first model's limited accuracy. It uses **VGG-style blocks of stacked 3 × 3 convolutions** while adding techniques intended to support more effective training and generalization.

**Architecture and training:**

- Three convolutional blocks, each with **two 3 × 3 convolutional layers**, using **32, 64, and 128** filters per block.
- **Batch normalization** after convolutional layers and the hidden dense layer.
- **Max pooling** and **dropout** (0.2, 0.3, and 0.4 in the convolutional blocks; 0.5 before the output layer).
- A **128-unit dense layer** and **10-class softmax** output.
- **He uniform initialization**, **SGD** optimizer (learning rate **0.001**, momentum **0.9**) and **categorical cross-entropy** loss.
- **Data augmentation:** random horizontal flips and width/height shifts up to 10%.
- **400 epochs**, batch size **64** for the augmented training generator.
- Pixel values scaled to **[0, 1]**; labels one-hot encoded for training.

**Recorded result:** **88.57% accuracy** on the standard 10,000-image CIFAR-10 test split.

This is a **custom VGG-inspired architecture trained from scratch**. It does **not** load pretrained VGG16/VGG19 weights or use transfer learning.

`GUI2.py` improves the interface by showing both the **predicted class** and a **preview of the selected image**.

## Results and evaluation

| | Initial CNN | VGG-inspired CNN |
| --- | --- | --- |
| Notebook | `CNN(2).ipynb` | `VGG(2).ipynb` |
| Training epochs | 20 | 400 |
| Optimization | Adam | SGD + momentum |
| Augmentation | Not used | Shifts and horizontal flips |
| Regularization | No dropout or batch normalization | Dropout and batch normalization |
| Evaluation images | 5,000 | 10,000 |
| **Recorded accuracy** | **73.00%** | **88.57%** |

Both notebooks also include **classification reports, confusion matrices, and training/validation plots** to examine model behavior beyond a single accuracy score.

### Evaluation note

These scores describe the outcomes recorded in the original notebooks, **not a controlled apples-to-apples benchmark**:

- The initial CNN evaluates on 5,000 images, with the other 5,000 images from the standard CIFAR-10 test split reserved for validation.
- The VGG-inspired notebook uses **the same 10,000 test images for both validation during training and the reported final evaluation**. Consequently, its reported accuracy is not from an independent, untouched final test set.
- The models also use different architectures, training schedules, and augmentation settings.

The second version achieved a higher **reported** classification accuracy, but a rigorous comparison would retrain and evaluate both models using the **same separate training, validation, and test protocol**.

## Classes and prediction workflow

The models predict one of these CIFAR-10 labels:

`plane` · `car` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

The desktop apps follow the same inference process:

1. Select an image from the computer.
2. Resize it to **32 × 32 pixels** and normalize pixel values by dividing by **255**.
3. Pass the image to the corresponding saved `.keras` model.
4. Select the class with the highest model output score and display its name.

The second interface additionally displays a **200 × 200 preview** of the selected image. For the existing GUI code, use an **RGB image** (for example, a standard JPEG); grayscale and RGBA images are not explicitly converted to RGB before prediction.

## Repository files

The following names match the files provided with this project:

```text
.
├── CNN(2).ipynb                  # Attempt 1: training, evaluation, plots
├── finalimageclassifier(2).keras # Attempt 1: saved model
├── GUI1.py                       # Attempt 1: basic Tkinter classifier
├── VGG(2).ipynb                  # Attempt 2: training, evaluation, plots
├── finalmodel(2).keras          # Attempt 2: saved model
├── GUI2.py                       # Attempt 2: Tkinter classifier + image preview
└── README.md
```

The dataset is **not** stored in the repository; the notebooks load CIFAR-10 through the Keras dataset API.

## Getting started

### 1. Install dependencies

**Python 3.10** is a suitable choice because it matches the version recorded in the notebooks. The saved models report **Keras 2.13.1** in their metadata; a TensorFlow/Keras 2.13 environment is a sensible starting point for compatibility.

From the project folder, create and activate a virtual environment if desired, then install:

```bash
python -m pip install tensorflow==2.13.1 numpy pillow opencv-python matplotlib seaborn scikit-learn notebook
```

The GUI scripts also use **Tkinter**, which is normally included with standard Python installations on Windows. On Linux, you may need your distribution's Tkinter package (often `python3-tk`).

### 2. Update the model paths in the GUIs

**Important:** The original GUI scripts contain absolute Windows file paths from the development computer. Before running them on another machine, replace those `load_model(...)` lines with paths to the model files in this repository.

Add this import near the top of **each** GUI file:

```python
from pathlib import Path
```

In **`GUI1.py`**, replace the existing model-loading line with:

```python
model = keras.models.load_model(
    Path(__file__).resolve().parent / "finalimageclassifier(2).keras",
    compile=False,
)
```

In **`GUI2.py`**, use:

```python
model = keras.models.load_model(
    Path(__file__).resolve().parent / "finalmodel(2).keras",
    compile=False,
)
```

`compile=False` is appropriate for **inference** and avoids needing to restore the original training optimizer and metrics. These paths assume the repository layout shown above. If you rename or move the files, adjust the paths accordingly.

### 3. Launch a classifier

To try the first version:

```bash
python GUI1.py
```

To try the second version:

```bash
python GUI2.py
```

Click **Open Image**, select a supported image, and view its predicted CIFAR-10 class. The second GUI also shows the image preview.

### 4. Explore or retrain the models (optional)

Open either notebook in Jupyter:

```bash
jupyter notebook
```

The notebooks contain the full data preparation, model construction, training, evaluation, and model-saving workflows. Before rerunning them, note that their example prediction cells reference **`../images/dog.jpg`**, and the saving cells write to **`../models/`**. Those locations reflect the original development folder layout and must be created or updated for your setup. The example `dog.jpg` is **not included** in the uploaded project files.

Retraining is **not required** to use the supplied saved models. Training the second model from scratch for 400 epochs can be computationally demanding.

## Possible next steps

- Evaluate both models on an identical, independent test set, with a separate validation split.
- Add early stopping and model checkpoints when retraining.
- Convert all incoming GUI images explicitly to RGB, validate file types, and show the predicted confidence score.
- Remove machine-specific file paths to make the project work immediately after cloning.

## Project context

This repository documents the **iterative development of a bachelor's senior-year milestone project**. Keeping both attempts makes the experimentation visible: an initial, functional CNN classifier followed by an architecture that introduces stacked convolutions, batch normalization, dropout, and data augmentation to achieve stronger recorded results on CIFAR-10.

**Dataset reference:** [CIFAR-10 — University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html)
