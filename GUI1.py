# %%
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.models.load_model('C:\\Users\\ENVY\\Downloads\\dev\\Project\\models\\finalimageclassifier.keras', compile=False)

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((32, 32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return prediction

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        prediction = predict_image(file_path)
        class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # Replace with your class names
        predicted_class = class_names[np.argmax(prediction)]
        label.config(text=f"Predicted class: {predicted_class}")

root = tk.Tk()
root.title("Image Classifier")

button = tk.Button(root, text="Open Image", command=open_image)
button.pack()

label = tk.Label(root, text="")
label.pack()

root.mainloop()
