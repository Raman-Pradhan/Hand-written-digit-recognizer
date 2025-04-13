import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tkinter as tk # this is for building the app
from PIL import Image, ImageTk, ImageOps 
from tkinter import filedialog #helps select any files.

#load the model(using the saved model.)
ANN_model = load_model(r"C:\Users\hp\Desktop\AIDS\Deep Learning\handwritten_digit_ann_model_best.h5")

#function to preprocess the image
def Image_preprocess(image_path):
    I = Image.open(image_path).convert('L') #reading the image, additiond a grayscale 
#resize the image optional step (optional step if the format is not matching)
    # I=I.resize((28,28))
    I = np.array(I)/255.0
    I=I.reshape(1,784)
    return I

def imageload_predict():
    file_path = filedialog.askopenfilename() #select the images by opening file explorer.
    #preprocess the image for display purpose 
    display_image = Image.open(file_path).resize((100,100))
    Tkinter_image = ImageTk.PhotoImage(display_image) #converting image to tkinter format
    image_panel.config(image = Tkinter_image)
    image_panel.image=Tkinter_image #to avoid garbage collection
    #pass the image for preprocessing
    Iprep = Image_preprocess(file_path)
    Prediction = ANN_model.predict(Iprep)
    Digit = np.argmax(Prediction)
    result_label.config(text = f'Predicted digit is: {Digit}') #configure result to canvas

# add the tkinter button and canvas
root = tk.Tk()
root.title("Digit Recognizer")
root.geometry("400x400") 
tk.Button(root, text = "Select digit image", command = imageload_predict).pack(pady=30) #give 30 pixel space vertically and then place button
image_panel = tk.Label(root)
image_panel.pack(pady = 10)
result_label = tk.Label(root, text = 'Predicted Digit = ', font=('Aerial', 16))
result_label.pack(pady = 40)

root.mainloop()