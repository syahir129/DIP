import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection & Recognition System (YOLO11)")
        self.root.geometry("1300x850")
        
        # --- 1. SETUP MODEL ---
        print("Loading YOLO11 model...")
        try:
            self.model = YOLO("yolo11n.pt")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")

        # Variables
        self.original_cv_image = None
        self.processed_cv_image = None
        self.detections = None 

        # --- 2. GUI LAYOUT ---
        self._setup_gui()

    def _setup_gui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # === LEFT CONTROL PANEL ===
        control_frame = tk.Frame(self.root, bg="#2c3e50", width=300, padx=15, pady=15)
        control_frame.grid(row=0, column=0, sticky="ns")
        control_frame.grid_propagate(False)

        tk.Label(control_frame, text="Control Panel", font=("Arial", 16, "bold"), bg="#2c3e50", fg="white").pack(pady=(0, 15))

        # 1. Load Image
        self.btn_style = {"font": ("Arial", 10), "bg": "#ecf0f1", "fg": "#2c3e50", "pady": 2, "width": 25}
        tk.Button(control_frame, text="1. Load Image", command=self.load_image, **self.btn_style).pack(pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # === ADJUSTABLE OPERATIONS ===
        
        # A. ENHANCEMENT
        tk.Label(control_frame, text="2. Enhancement (Sharpen)", font=("Arial", 10, "bold"), bg="#2c3e50", fg="#bdc3c7").pack(anchor="w")
        self.scale_sharp = tk.Scale(control_frame, from_=0, to=100, orient="horizontal", bg="#2c3e50", fg="white", label="Intensity %")
        self.scale_sharp.set(50) # Default
        self.scale_sharp.pack(fill="x")
        tk.Button(control_frame, text="Apply Sharpen", command=self.apply_enhancement_roi, **self.btn_style).pack(pady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # B. COLOR PROCESS
        tk.Label(control_frame, text="3. Color Process (Saturation)", font=("Arial", 10, "bold"), bg="#2c3e50", fg="#bdc3c7").pack(anchor="w")
        self.scale_color = tk.Scale(control_frame, from_=0, to=100, orient="horizontal", bg="#2c3e50", fg="white", label="Add Saturation")
        self.scale_color.set(40) # Default
        self.scale_color.pack(fill="x")
        tk.Button(control_frame, text="Apply Color", command=self.apply_color_roi, **self.btn_style).pack(pady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # C. SEGMENTATION
        tk.Label(control_frame, text="4. Segmentation (Threshold)", font=("Arial", 10, "bold"), bg="#2c3e50", fg="#bdc3c7").pack(anchor="w")
        self.scale_thresh = tk.Scale(control_frame, from_=0, to=255, orient="horizontal", bg="#2c3e50", fg="white", label="Threshold Value")
        self.scale_thresh.set(127) # Default
        self.scale_thresh.pack(fill="x")
        tk.Button(control_frame, text="Apply Segmentation", command=self.apply_segmentation_roi, **self.btn_style).pack(pady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # D. DETECTION SETTINGS
        tk.Label(control_frame, text="AI Settings", font=("Arial", 10, "bold"), bg="#2c3e50", fg="#bdc3c7").pack(anchor="w")
        self.scale_conf = tk.Scale(control_frame, from_=10, to=100, orient="horizontal", bg="#2c3e50", fg="white", label="Confidence %")
        self.scale_conf.set(25) # Default 25%
        self.scale_conf.pack(fill="x")
        tk.Button(control_frame, text="Re-Detect Objects", command=self.force_redetect, bg="#e74c3c", fg="white", font=("Arial", 10, "bold"), pady=5, width=25).pack(pady=5)
        
        # Reset
        tk.Button(control_frame, text="Reset Image", command=self.reset_image, bg="#95a5a6", fg="white", width=25).pack(side="bottom", pady=20)

        # === RIGHT DISPLAY AREA ===
        display_frame = tk.Frame(self.root, bg="#ecf0f1")
        display_frame.grid(row=0, column=1, sticky="nsew")
        
        self.lbl_image = tk.Label(display_frame, text="No Image Loaded", bg="#bdc3c7", fg="#7f8c8d", font=("Arial", 20))
        self.lbl_image.place(relx=0.5, rely=0.5, anchor="center")
        
        self.lbl_status = tk.Label(display_frame, text="System Ready", bg="#34495e", fg="white", anchor="w", padx=10)
        self.lbl_status.pack(side="bottom", fill="x")

    # --- FUNCTIONS ---

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path:
            self.original_cv_image = cv2.imread(path)
            self.processed_cv_image = self.original_cv_image.copy()
            self.detections = None 
            self.show_image(self.original_cv_image)
            self.lbl_status.config(text=f"Loaded: {path}")

    def ensure_detections(self):
        """Run YOLO if detections are missing or if we need to update confidence."""
        if self.detections is None:
            self.force_redetect()
            if len(self.detections) == 0:
                return False
        return True

    def force_redetect(self):
        """Manually runs YOLO with the selected confidence slider."""
        if self.original_cv_image is None: return
        
        conf_val = self.scale_conf.get() / 100.0  # Convert 0-100 to 0.0-1.0
        
        self.lbl_status.config(text=f"Detecting objects (Conf: {conf_val})...")
        self.root.update()
        
        results = self.model(self.original_cv_image, conf=conf_val)
        self.detections = results[0].boxes
        
        # Show the raw detections immediately
        res_plotted = results[0].plot()
        self.processed_cv_image = res_plotted
        self.show_image(self.processed_cv_image)
        self.lbl_status.config(text=f"Detection Complete: {len(self.detections)} objects found.")

    def apply_enhancement_roi(self):
        """Blends sharpened image with original based on slider."""
        if self.original_cv_image is None or not self.ensure_detections(): return
        
        temp_img = self.original_cv_image.copy()
        intensity = self.scale_sharp.get() / 100.0 # 0.0 to 1.0
        
        for box in self.detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = temp_img[y1:y2, x1:x2]
            
            # Create sharpened version
            kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
            sharpened = cv2.filter2D(roi, -1, kernel)
            
            # Blend: output = (1-alpha)*original + alpha*sharpened
            blended = cv2.addWeighted(roi, 1.0 - intensity, sharpened, intensity, 0)
            
            # Draw green box
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            temp_img[y1:y2, x1:x2] = blended
            
        self.processed_cv_image = temp_img
        self.show_image(self.processed_cv_image)
        self.lbl_status.config(text=f"Applied: Sharpening (Intensity: {int(intensity*100)}%)")

    def apply_color_roi(self):
        """Boosts saturation by slider amount."""
        if self.original_cv_image is None or not self.ensure_detections(): return
        
        temp_img = self.original_cv_image.copy()
        sat_boost = self.scale_color.get()
        
        for box in self.detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = temp_img[y1:y2, x1:x2]
            
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Add saturation with clipping to 255
            s = cv2.add(s, sat_boost)
            
            final_hsv = cv2.merge((h, s, v))
            processed_roi = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            temp_img[y1:y2, x1:x2] = processed_roi

        self.processed_cv_image = temp_img
        self.show_image(self.processed_cv_image)
        self.lbl_status.config(text=f"Applied: Color Saturation (+{sat_boost})")

    def apply_segmentation_roi(self):
        """Manual Thresholding based on slider."""
        if self.original_cv_image is None or not self.ensure_detections(): return
        
        temp_img = self.original_cv_image.copy()
        thresh_val = self.scale_thresh.get()
        
        for box in self.detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = temp_img[y1:y2, x1:x2]
            
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Manual Binary Thresholding using slider value
            _, mask = cv2.threshold(gray_roi, thresh_val, 255, cv2.THRESH_BINARY)
            
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            temp_img[y1:y2, x1:x2] = mask_bgr

        self.processed_cv_image = temp_img
        self.show_image(self.processed_cv_image)
        self.lbl_status.config(text=f"Applied: Manual Threshold (Value: {thresh_val})")

    def reset_image(self):
        if self.original_cv_image is not None:
            self.processed_cv_image = self.original_cv_image.copy()
            self.show_image(self.processed_cv_image)
            self.detections = None
            self.lbl_status.config(text="Image Reset")

    def show_image(self, cv_img):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(rgb_img)
        display_w, display_h = 900, 700
        im_pil.thumbnail((display_w, display_h))
        imgtk = ImageTk.PhotoImage(image=im_pil)
        self.lbl_image.configure(image=imgtk, text="")
        self.lbl_image.image = imgtk

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()