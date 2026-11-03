# %%
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.models.load_model('C:\\Users\\ENVY\\Downloads\\dev\\Project\\models\\finalmodel.keras')

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((32, 32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img, verbose=0)
    return prediction

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        prediction = predict_image(file_path)
        class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        predicted_class = class_names[np.argmax(prediction)]
        label.config(text=f"Predicted class: {predicted_class}")
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

root = tk.Tk()
root.title("Image Classifier")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

open_button = ttk.Button(frame, text="Open Image", command=open_image)
open_button.grid(row=0, column=0, pady=5)

label = ttk.Label(frame, text="")
label.grid(row=1, column=0, pady=5)

image_label = ttk.Label(frame)
image_label.grid(row=2, column=0, pady=5)

root.mainloop()
