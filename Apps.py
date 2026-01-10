import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Set modern theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ModernApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("DIP Intelligent System (Assignment 1)")
        self.geometry("1400x900")
        
        # Grid Layout (2 Columns: Sidebar, Main Display)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- AI SETUP ---
        self.model = None
        self.original_cv = None
        self.processed_cv = None
        self.detections = None
        self._init_model()

        # --- UI COMPONENTS ---
        self._setup_sidebar()
        self._setup_main_area()

    def _init_model(self):
        print("Loading YOLO11...")
        try:
            self.model = YOLO("yolo11n.pt")
        except Exception as e:
            print(f"Error: {e}")

    def _setup_sidebar(self):
        # Sidebar Frame
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(9, weight=1) # Spacer at bottom

        # Logo / Title
        logo = ctk.CTkLabel(self.sidebar, text="DIP SYSTEM", font=ctk.CTkFont(size=24, weight="bold"))
        logo.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # 1. Load Image
        self.btn_load = ctk.CTkButton(self.sidebar, text="📂 Load Image", command=self.load_image, fg_color="#2ecc71", hover_color="#27ae60")
        self.btn_load.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        # Separator
        ctk.CTkLabel(self.sidebar, text="ASSIGNMENT MODULES", text_color="gray").grid(row=2, column=0, pady=(10,0))

        # --- MODULE BUTTONS ---
        # e. Pattern Recognition (Primary)
        self.frame_rec = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.frame_rec.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.frame_rec, text="(e) Pattern Recog.", font=("Arial", 12, "bold")).pack(anchor="w")
        self.slider_conf = ctk.CTkSlider(self.frame_rec, from_=10, to=100, number_of_steps=90)
        self.slider_conf.set(30)
        self.slider_conf.pack(fill="x", pady=5)
        
        self.btn_rec = ctk.CTkButton(self.frame_rec, text="▶ Run AI Recognition", command=self.run_recognition, fg_color="#e74c3c", hover_color="#c0392b")
        self.btn_rec.pack(fill="x")

        # Other Modules
        self.btn_enhance = ctk.CTkButton(self.sidebar, text="(a) Enhancement (Sharpen)", command=self.run_enhance)
        self.btn_enhance.grid(row=4, column=0, padx=20, pady=5, sticky="ew")

        self.btn_color = ctk.CTkButton(self.sidebar, text="(b) Color (Saturation)", command=self.run_color)
        self.btn_color.grid(row=5, column=0, padx=20, pady=5, sticky="ew")

        self.btn_seg = ctk.CTkButton(self.sidebar, text="(c) Segmentation (Thresh)", command=self.run_segmentation)
        self.btn_seg.grid(row=6, column=0, padx=20, pady=5, sticky="ew")

        self.btn_feat = ctk.CTkButton(self.sidebar, text="(d) Feature Ext. (Edges)", command=self.run_features)
        self.btn_feat.grid(row=7, column=0, padx=20, pady=5, sticky="ew")

        # Reset
        self.btn_reset = ctk.CTkButton(self.sidebar, text="↺ Reset Image", command=self.reset_image, fg_color="gray", hover_color="#555")
        self.btn_reset.grid(row=8, column=0, padx=20, pady=20, sticky="ew")

    def _setup_main_area(self):
        # Dashboard Frame
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # Status Bar (Top)
        self.status_label = ctk.CTkLabel(self.main_frame, text="Status: Ready", font=("Consolas", 14), anchor="w")
        self.status_label.grid(row=0, column=0, padx=20, pady=(10,0), sticky="ew")

        # Image Display Area
        self.image_label = ctk.CTkLabel(self.main_frame, text="[ No Image Loaded ]", font=("Arial", 20), text_color="gray")
        self.image_label.grid(row=1, column=0, padx=10, pady=10)

    # --- LOGIC ---

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.webp")])
        if path:
            self.original_cv = cv2.imread(path)
            if self.original_cv is None: return
            self.reset_image()
            self.status_label.configure(text=f"Loaded: {path.split('/')[-1]}")

    def ensure_ai(self):
        if self.original_cv is None:
            messagebox.showwarning("Warning", "Load an image first!")
            return False
        if self.detections is None:
            self.run_recognition()
        return self.detections is not None

    # (e) Pattern Recognition
    def run_recognition(self):
        if self.original_cv is None: return
        conf = self.slider_conf.get() / 100.0
        
        self.status_label.configure(text=f"AI Thinking (Conf: {conf})...")
        self.update()
        
        res = self.model(self.original_cv, conf=conf)
        self.detections = res[0].boxes
        
        self.processed_cv = res[0].plot()
        self.display_image(self.processed_cv)
        self.status_label.configure(text=f"Pattern Recognition: Found {len(self.detections)} objects")

    # (a) Enhancement
    def run_enhance(self):
        if not self.ensure_ai(): return
        img = self.original_cv.copy()
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) # Sharpen
        
        for box in self.detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = img[y1:y2, x1:x2]
            img[y1:y2, x1:x2] = cv2.filter2D(roi, -1, kernel)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        self.display_image(img)
        self.status_label.configure(text="Applied: Image Sharpening")

    # (b) Color Processing
    def run_color(self):
        if not self.ensure_ai(): return
        img = self.original_cv.copy()
        
        for box in self.detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = img[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = cv2.add(s, 50) # Add Saturation
            img[y1:y2, x1:x2] = cv2.cvtColor(cv2.merge((h,s,v)), cv2.COLOR_HSV2BGR)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

        self.display_image(img)
        self.status_label.configure(text="Applied: Color Saturation Boost")

    # (c) Segmentation
    def run_segmentation(self):
        if not self.ensure_ai(): return
        img = self.original_cv.copy()
        
        for box in self.detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            img[y1:y2, x1:x2] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        self.display_image(img)
        self.status_label.configure(text="Applied: Binary Segmentation")

    # (d) Feature Extraction
    def run_features(self):
        if not self.ensure_ai(): return
        img = self.original_cv.copy()
        
        for box in self.detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = img[y1:y2, x1:x2]
            edges = cv2.Canny(roi, 100, 200)
            img[y1:y2, x1:x2] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        self.display_image(img)
        self.status_label.configure(text="Applied: Canny Edge Features")

    def reset_image(self):
        if self.original_cv is not None:
            self.detections = None
            self.display_image(self.original_cv)
            self.status_label.configure(text="Image Reset")

    def display_image(self, cv_img):
        # Convert to RGB for Pillow
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        # Calculate aspect ratio
        w, h = 900, 700
        # FIXED: Use ctk.CTkImage instead of importing it from PIL
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(w, h))
        
        self.image_label.configure(image=ctk_img, text="")

if __name__ == "__main__":
    app = ModernApp()
    app.mainloop()