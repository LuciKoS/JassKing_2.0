#All necessary imporst

from ultralytics import YOLO
import joblib
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import joblib
import os
from PIL import Image, ImageTk

#Initializations

image_model = YOLO('ImageRecognitionModel/weights/best.pt')
trumpf_model = joblib.load('rf_model_new.joblib')

cards_nums = {
    "Ei_6":0,
    "Ei_7":1,
    "Ei_8":2,
    "Ei_9":3,
    "Ei_10":4,
    "Ei_U":5,
    "Ei_O":6,
    "Ei_K":7,
    "Ei_A":8,
    "Ro_6":9,
    "Ro_7":10,
    "Ro_8":11,
    "Ro_9":12,
    "Ro_10":13,
    "Ro_U":14,
    "Ro_O":15,
    "Ro_K":16,
    "Ro_A":17,
    "Se_6": 18,
    "Se_7": 19,
    "Se_8": 20,
    "Se_9": 21,
    "Se_10": 22,
    "Se_U": 23,
    "Se_O": 24,
    "Se_K": 25,
    "Se_A": 26,
    "Si_6": 27,
    "Si_7": 28,
    "Si_8": 29,
    "Si_9": 30,
    "Si_10": 31,
    "Si_U": 32,
    "Si_O": 33,
    "Si_K": 34,
    "Si_A": 35,
}

#Creating App Class

class JassKingApp():
    def __init__(self, master):

        self.master = master

        master.title("Jass King App")

        master.geometry("800x600")

        self.image_label = tk.Label(master)
        self.image_label.pack(pady = 10)

        self.upload_button = tk.Button(master, text = "Upload your hand", command = self.upload_image)
        self.upload_button.pack(pady = 10)





    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title = "Select an image pls",
            filetypes=[("Image Files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"))]
        )

        if file_path:
            img = Image.open(file_path)

            max_size = (300, 300)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            self.image_label.config(image = tk_img)
            self.image_label.image = tk_img



if __name__ == "__main__":
    root = tk.Tk()
    app = JassKingApp(root)
    root.mainloop()