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

cards_images = {
    "Ei_6": "card_images/img_0.jpg",
    "Ei_7": "card_images/img_1.jpg",
    "Ei_8": "card_images/img_2.jpg",
    "Ei_9": "card_images/img_3.jpg",
    "Ei_10": "card_images/img_4.jpg",
    "Ei_U": "card_images/img_5.jpg",
    "Ei_O": "card_images/img_6.jpg",
    "Ei_K": "card_images/img_7.jpg",
    "Ei_A": "card_images/img_8.jpg",
    "Ro_6": "card_images/img_9.jpg",
    "Ro_7": "card_images/img_10.jpg",
    "Ro_8": "card_images/img_11.jpg",
    "Ro_9": "card_images/img_12.jpg",
    "Ro_10": "card_images/img_13.jpg",
    "Ro_U": "card_images/img_14.jpg",
    "Ro_O": "card_images/img_15.jpg",
    "Ro_K": "card_images/img_16.jpg",
    "Ro_A": "card_images/img_17.jpg",
    "Se_6": "card_images/img_18.jpg",
    "Se_7": "card_images/img_19.jpg",
    "Se_8": "card_images/img_20.jpg",
    "Se_9": "card_images/img_21.jpg",
    "Se_10": "card_images/img_22.jpg",
    "Se_U": "card_images/img_23.jpg",
    "Se_O": "card_images/img_24.jpg",
    "Se_K": "card_images/img_25.jpg",
    "Se_A": "card_images/img_26.jpg",
    "Si_6": "card_images/img_27.jpg",
    "Si_7": "card_images/img_28.jpg",
    "Si_8": "card_images/img_29.jpg",
    "Si_9": "card_images/img_30.jpg",
    "Si_10": "card_images/img_31.jpg",
    "Si_U": "card_images/img_32.jpg",
    "Si_O": "card_images/img_33.jpg",
    "Si_K": "card_images/img_34.jpg",
    "Si_A": "card_images/img_35.jpg",
}

#Creating App Class

class JassKingApp():
    def __init__(self, master):
        self.master = master
        master.title("Jass King App")
        master.geometry("800x600")
        
        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height)
        self.canvas.place(relx=0.5, rely=0.5, anchor="center")
        
        bg_image = Image.open("bg3.png")
        bg_image = bg_image.resize((self.canvas_width, self.canvas_height), Image.Resampling.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(bg_image)
        self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")
        
        # Upload button below the canvas
        self.upload_button = tk.Button(master, text="Upload your hand", command=self.upload_image)
        self.upload_button.pack(pady=10)
        
        self.card_items = []
        self.card_photos = []

    def upload_image(self):
        # Clear previous card images from the canvas
        for item in self.card_items:
            self.canvas.delete(item)
        self.card_items.clear()
        self.card_photos.clear()
        
        file_path = filedialog.askopenfilename(
            title="Select an image please",
            filetypes=[("Image Files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"))]
        )
        
        if file_path:
            img = Image.open(file_path)
            
            cards = image_model(img, imgsz=640, conf=0.3)
            detections = []
            for card in cards:
                for box in card.boxes:
                    conf = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)
                    cls_id = int(box.cls[0]) if hasattr(box.cls, '__len__') else int(box.cls)
                    label = image_model.model.names[cls_id]
                    detections.append([label, conf])
            detections.sort(key=lambda x: x[1], reverse=True)
            top_detections = detections[:9]
            
            mapped_cards = [cards_nums.get(card[0], -1) for card in top_detections]
            
            if len(mapped_cards) == 9:
                try:
                    cards_array = np.array(mapped_cards).reshape(1, -1)
                    trumpf_prediction = trumpf_model.predict(cards_array)
                    messagebox.showinfo("Trumpf Model Prediction", f"Prediction: {trumpf_prediction}")
                except Exception as e:
                    messagebox.showerror("Error", f"Prediction failed: {e}")
            else:
                messagebox.showwarning("Invalid Card Detection", "Not all detected cards are valid for trumpf prediction.")
            
            card_width = 100
            card_height = 150
            card_spacing = 10

            num_cards_top = 5
            top_x_start = (self.canvas_width - (num_cards_top * card_width + (num_cards_top - 1) * card_spacing)) / 2
            top_y_position = self.canvas_height / 2 - card_height - 10

            num_cards_bottom = 4
            bottom_x_start = (self.canvas_width - (num_cards_bottom * card_width + (num_cards_bottom - 1) * card_spacing)) / 2
            bottom_y_position = self.canvas_height / 2 + 10

            for i, (card, conf) in enumerate(top_detections):
                if i < num_cards_top:
                    x = top_x_start + i * (card_width + card_spacing)
                    y = top_y_position
                else:
                    x = bottom_x_start + (i - num_cards_top) * (card_width + card_spacing)
                    y = bottom_y_position

                img_path = cards_images.get(card)
                if img_path and os.path.exists(img_path):
                    card_img = Image.open(img_path)
                    card_img = card_img.resize((card_width, card_height), Image.Resampling.LANCZOS)
                    tk_card_img = ImageTk.PhotoImage(card_img)
                    self.card_photos.append(tk_card_img)
                    card_item = self.canvas.create_image(x, y, image=tk_card_img, anchor="nw")
                    self.card_items.append(card_item)
                else:
                    print(f"Image path not found for card {card}")

if __name__ == "__main__":
    root = tk.Tk()
    app = JassKingApp(root)
    root.mainloop()