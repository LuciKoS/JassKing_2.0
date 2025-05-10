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
        master.geometry("600x700")  # Reduced window size
        
        # Add a frame for the uploaded image at the very top
        self.uploaded_image_frame = tk.Frame(master)
        self.uploaded_image_frame.pack(side=tk.TOP, pady=5)  # Reduced padding
        self.uploaded_image_label = tk.Label(self.uploaded_image_frame)
        self.uploaded_image_label.pack()
        
        # Store the uploaded image
        self.uploaded_image = None
        self.uploaded_image_photo = None
        
        # Upload button below the image
        self.upload_button = tk.Button(master, text="Upload your hand", command=self.upload_image)
        self.upload_button.pack(pady=5)  # Reduced padding
        
        # Canvas for cards below the upload button
        self.canvas_width = 600  # Reduced canvas width
        self.canvas_height = 400  # Reduced canvas height
        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(pady=5)  # Reduced padding
        
        bg_image = Image.open("bg3.png")
        bg_image = bg_image.resize((self.canvas_width, self.canvas_height), Image.Resampling.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(bg_image)
        self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")
        
        self.card_items = []
        self.card_photos = []
        self.remove_buttons = []  # Store references to remove buttons
        
        # Add buttons at the bottom
        self.button_frame = tk.Frame(master)
        self.button_frame.pack(side=tk.BOTTOM, pady=10)
        
        self.add_card_button = tk.Button(self.button_frame, text="Add a Card", command=self.add_card)
        self.add_card_button.pack(side=tk.LEFT, padx=5)
        
        self.predict_button = tk.Button(self.button_frame, text="Predict Again", command=self.predict_cards)
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        # Store selected cards
        self.selected_cards = []

    def add_card(self):
        # Create a new window for card selection
        card_window = tk.Toplevel(self.master)
        card_window.title("Select a Card")
        
        # Create a canvas with scrollbar for the cards
        canvas = tk.Canvas(card_window)
        scrollbar = tk.Scrollbar(card_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add all cards to the scrollable frame
        row = 0
        col = 0
        for card_name, img_path in cards_images.items():
            if os.path.exists(img_path):
                card_img = Image.open(img_path)
                card_img = card_img.resize((80, 120), Image.Resampling.LANCZOS)  # Reduced card size
                tk_card_img = ImageTk.PhotoImage(card_img)
                
                # Keep reference to prevent garbage collection
                if not hasattr(self, 'card_selection_photos'):
                    self.card_selection_photos = []
                self.card_selection_photos.append(tk_card_img)
                
                btn = tk.Button(scrollable_frame, image=tk_card_img, 
                              command=lambda c=card_name: self.select_card(c, card_window))
                btn.grid(row=row, column=col, padx=5, pady=5)
                
                col += 1
                if col > 3:  # 4 cards per row
                    col = 0
                    row += 1
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def select_card(self, card_name, window):
        if len(self.selected_cards) < 9:
            self.selected_cards.append(card_name)
            self.display_cards()
            window.destroy()
        else:
            messagebox.showwarning("Warning", "Maximum 9 cards allowed!")

    def display_cards(self):
        # Clear previous card images and remove buttons
        for item in self.card_items:
            self.canvas.delete(item)
        for btn in self.remove_buttons:
            btn.destroy()
        
        self.card_items.clear()
        self.card_photos.clear()
        self.remove_buttons.clear()
        
        card_width = 80
        card_height = 120
        card_spacing = 8

        num_cards_top = 5
        top_x_start = (self.canvas_width - (num_cards_top * card_width + (num_cards_top - 1) * card_spacing)) / 2
        top_y_position = self.canvas_height / 2 - card_height - 5

        num_cards_bottom = 4
        bottom_x_start = (self.canvas_width - (num_cards_bottom * card_width + (num_cards_bottom - 1) * card_spacing)) / 2
        bottom_y_position = self.canvas_height / 2 + 5

        # Create a list of 9 cards, padding with None if needed
        display_cards = self.selected_cards.copy()
        while len(display_cards) < 9:
            display_cards.append(None)

        for i, card in enumerate(display_cards):
            if i < num_cards_top:
                x = top_x_start + i * (card_width + card_spacing)
                y = top_y_position
            else:
                x = bottom_x_start + (i - num_cards_top) * (card_width + card_spacing)
                y = bottom_y_position

            if card is not None:  # If we have a real card
                img_path = cards_images.get(card)
                if img_path and os.path.exists(img_path):
                    card_img = Image.open(img_path)
                    card_img = card_img.resize((card_width, card_height), Image.Resampling.LANCZOS)
                    tk_card_img = ImageTk.PhotoImage(card_img)
                    self.card_photos.append(tk_card_img)
                    card_item = self.canvas.create_image(x, y, image=tk_card_img, anchor="nw")
                    self.card_items.append(card_item)
                    
                    # Create remove button for each card
                    remove_btn = tk.Button(self.canvas, text="X", 
                                         command=lambda idx=i: self.remove_card(idx),
                                         bg='red', fg='white', width=1)
                    remove_btn.place(x=x + card_width - 15, y=y + 3)
                    self.remove_buttons.append(remove_btn)
            else:  # If we need to display an empty card
                # Create a blank card (gray rectangle)
                card_item = self.canvas.create_rectangle(
                    x, y, x + card_width, y + card_height,
                    fill='gray', outline='black'
                )
                self.card_items.append(card_item)
                
                # Add a plus button for empty cards
                add_btn = tk.Button(self.canvas, text="+", 
                                  command=self.add_card,
                                  bg='green', fg='white', width=1)
                add_btn.place(x=x + card_width - 15, y=y + 3)
                self.remove_buttons.append(add_btn)

    def remove_card(self, index):
        if 0 <= index < len(self.selected_cards):
            self.selected_cards.pop(index)
            self.display_cards()

    def predict_cards(self):
        if len(self.selected_cards) == 9:
            try:
                mapped_cards = [cards_nums.get(card, -1) for card in self.selected_cards]
                # Create a list of 17 elements with commas between card numbers
                full_input = []
                for i, card in enumerate(mapped_cards):
                    full_input.append(card)
                    if i < 8:  # Add comma after each card except the last one
                        full_input.append(17)  # 17 represents the comma in the model
                
                cards_array = np.array(full_input).reshape(1, -1)
                
                # Debug print
                print("Input array shape:", cards_array.shape)
                print("Input array:", cards_array)
                
                # Get prediction probabilities
                trumpf_probs = trumpf_model.predict_proba(cards_array)[0]
                
                # Debug print
                print("Probabilities shape:", trumpf_probs.shape)
                print("Probabilities:", trumpf_probs)
                
                # Get top 3 predictions with their probabilities
                top_3_indices = np.argsort(trumpf_probs)[-3:][::-1]  # Get indices of top 3 probabilities
                top_3_predictions = []
                
                # Map indices to trump names (assuming 0=Eichel, 1=Rose, 2=Schilten, 3=Schelle)
                trump_names = ["Eichel", "Rose", "Schilten", "Schelle"]
                
                for idx in top_3_indices:
                    if idx < len(trump_names):  # Check if index is valid
                        probability = trumpf_probs[idx] * 100  # Convert to percentage
                        top_3_predictions.append(f"{trump_names[idx]}: {probability:.1f}%")
                
                # Create message with top 3 predictions
                if top_3_predictions:
                    message = "Top 3 Trump Predictions:\n\n" + "\n".join(top_3_predictions)
                    messagebox.showinfo("Trumpf Model Prediction", message)
                else:
                    messagebox.showerror("Error", "No valid predictions were generated")
                
            except Exception as e:
                print("Error details:", str(e))  # Debug print
                messagebox.showerror("Error", f"Prediction failed: {str(e)}\nPlease check the console for details.")
        else:
            messagebox.showwarning("Invalid Card Selection", "Please select exactly 9 cards for prediction.")

    def upload_image(self):
        # Clear previous card images and remove buttons
        for item in self.card_items:
            self.canvas.delete(item)
        for btn in self.remove_buttons:
            btn.destroy()
        self.card_items.clear()
        self.card_photos.clear()
        self.remove_buttons.clear()
        
        file_path = filedialog.askopenfilename(
            title="Select an image please",
            filetypes=[("Image Files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"))]
        )
        
        if file_path:
            # Display the uploaded image
            img = Image.open(file_path)
            img.thumbnail((300, 200))
            self.uploaded_image = img
            self.uploaded_image_photo = ImageTk.PhotoImage(img)
            self.uploaded_image_label.configure(image=self.uploaded_image_photo)
            
            cards = image_model(img, imgsz=640, conf=0.3)
            detections = []
            seen_cards = set()  # Keep track of unique cards
            
            for card in cards:
                for box in card.boxes:
                    conf = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)
                    cls_id = int(box.cls[0]) if hasattr(box.cls, '__len__') else int(box.cls)
                    label = image_model.model.names[cls_id]
                    
                    # Only add card if we haven't seen it before
                    if label not in seen_cards:
                        detections.append([label, conf])
                        seen_cards.add(label)
            
            # Sort by confidence and take top 9 unique cards
            detections.sort(key=lambda x: x[1], reverse=True)
            top_detections = detections[:9]
            
            # Update selected cards with detected cards
            self.selected_cards = [card[0] for card in top_detections]
            self.display_cards()
            
            card_width = 80  # Reduced card width
            card_height = 120  # Reduced card height
            card_spacing = 8  # Reduced spacing

            num_cards_top = 5
            top_x_start = (self.canvas_width - (num_cards_top * card_width + (num_cards_top - 1) * card_spacing)) / 2
            top_y_position = self.canvas_height / 2 - card_height - 5  # Reduced spacing

            num_cards_bottom = 4
            bottom_x_start = (self.canvas_width - (num_cards_bottom * card_width + (num_cards_bottom - 1) * card_spacing)) / 2
            bottom_y_position = self.canvas_height / 2 + 5  # Reduced spacing

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