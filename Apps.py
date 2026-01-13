import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import random

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO11 Instance Segmentation (Exact Shape Detection)")
        self.root.geometry("1500x950")
        
        # --- 1. SETUP SEGMENTATION MODEL ---
        print("Loading YOLO11 Segmentation model...")
        try:
            # CHANGED: 'seg' model provides masks (shapes) instead of just boxes
            self.model = YOLO("yolo11n-seg.pt") 
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")

        self.original_cv_image = None
        self.current_results = None 
        self.cap = None
        self.is_camera_on = False

        self._setup_gui()

    def _setup_gui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # === LEFT CONTROL PANEL ===
        control_frame = tk.Frame(self.root, bg="#2c3e50", width=400, padx=15, pady=15)
        control_frame.grid(row=0, column=0, sticky="ns")
        control_frame.grid_propagate(False)

        tk.Label(control_frame, text="DIP Control Panel", font=("Arial", 16, "bold"), bg="#2c3e50", fg="white").pack(pady=(0, 15))

        # Media Inputs
        self.btn_style = {"font": ("Arial", 9), "bg": "#ecf0f1", "fg": "#2c3e50", "pady": 2, "width": 35}
        tk.Button(control_frame, text="📁 Load Image", command=self.load_image, **self.btn_style).pack(pady=2)
        self.btn_camera = tk.Button(control_frame, text="📷 Open Camera", command=self.toggle_camera, bg="#2ecc71", fg="white", font=("Arial", 9, "bold"), width=35)
        self.btn_camera.pack(pady=2)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # --- A. ENHANCEMENT ---
        tk.Label(control_frame, text="a. Enhancement (Ch 3-5)", font=("Arial", 9, "bold"), bg="#2c3e50", fg="#bdc3c7").pack(anchor="w", pady=(5, 0))
        self.enhance_mode = tk.StringVar(value="Sharpening")
        self.combo_enhance = ttk.Combobox(control_frame, textvariable=self.enhance_mode, state="readonly", width=33)
        self.combo_enhance['values'] = ("Sharpening", "Histogram Equalization", "Gamma Correction")
        self.combo_enhance.pack(pady=2)
        self.scale_enhance = tk.Scale(control_frame, from_=0, to=100, orient="horizontal", bg="#2c3e50", fg="white", length=300, label="Intensity")
        self.scale_enhance.set(50)
        self.scale_enhance.pack()
        tk.Button(control_frame, text="Apply Enhancement", command=self.apply_enhancement, **self.btn_style).pack(pady=(0, 5))

        # --- B. COLOUR (SHAPE FILL) ---
        tk.Label(control_frame, text="b. Colour (Fill Exact Shape)", font=("Arial", 9, "bold"), bg="#2c3e50", fg="#bdc3c7").pack(anchor="w", pady=(5, 0))
        self.scale_color = tk.Scale(control_frame, from_=0, to=100, orient="horizontal", bg="#2c3e50", fg="white", length=300, label="Opacity %")
        self.scale_color.set(50)
        self.scale_color.pack()
        tk.Button(control_frame, text="Apply Random Color Fill", command=self.apply_shape_color, **self.btn_style).pack(pady=(0, 5))

        # --- C. SEGMENTATION DISPLAY ---
        tk.Label(control_frame, text="c. Segmentation Mask", font=("Arial", 9, "bold"), bg="#2c3e50", fg="#bdc3c7").pack(anchor="w", pady=(5, 0))
        tk.Button(control_frame, text="Isolate Masks (Black BG)", command=self.apply_mask_isolation, **self.btn_style).pack(pady=2)

        # --- D. FEATURE EXTRACTION ---
        tk.Label(control_frame, text="d. Feature Extraction", font=("Arial", 9, "bold"), bg="#2c3e50", fg="#bdc3c7").pack(anchor="w", pady=(5, 0))
        tk.Button(control_frame, text="Draw Contours (Outlines)", command=self.apply_contours, **self.btn_style).pack(pady=2)

        # --- E. PATTERN RECOGNITION ---
        tk.Label(control_frame, text="e. Pattern Recognition", font=("Arial", 9, "bold"), bg="#2c3e50", fg="#bdc3c7").pack(anchor="w", pady=(10,0))
        tk.Button(control_frame, text="Show YOLO Segmentation", command=self.apply_pattern_recognition, bg="#e67e22", fg="white", font=("Arial", 9, "bold"), width=35).pack(pady=5)

        # Reset
        tk.Button(control_frame, text="Reset Image", command=self.reset_image, bg="#95a5a6", fg="white", width=35).pack(side="bottom", pady=10)

        # === DISPLAY ===
        display_frame = tk.Frame(self.root, bg="#ecf0f1")
        display_frame.grid(row=0, column=1, sticky="nsew")
        self.lbl_image = tk.Label(display_frame, text="Ready", bg="#bdc3c7", font=("Arial", 18))
        self.lbl_image.place(relx=0.5, rely=0.5, anchor="center")
        self.lbl_status = tk.Label(display_frame, text="Idle", bg="#34495e", fg="white", anchor="w", padx=10)
        self.lbl_status.pack(side="bottom", fill="x")

    # --- LOGIC ---
    
    def ensure_data(self):
        if self.original_cv_image is None: return False
        # Run inference if we haven't already
        if self.current_results is None:
            self.current_results = self.model(self.original_cv_image, conf=0.25, verbose=False)[0]
        return True

    def apply_enhancement(self):
        """Standard enhancement applied to the WHOLE image for simplicity."""
        if self.original_cv_image is None: return
        temp = self.original_cv_image.copy()
        mode = self.enhance_mode.get()
        val = self.scale_enhance.get() / 100.0
        
        if "Sharpening" in mode:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharp = cv2.filter2D(temp, -1, kernel)
            temp = cv2.addWeighted(temp, 1.0 - val, sharp, val, 0)
        elif "Histogram" in mode:
            yuv = cv2.cvtColor(temp, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            temp = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        self.show_image(temp)
        self.lbl_status.config(text=f"Applied Enhancement: {mode}")

    def apply_shape_color(self):
        """
        Fills the EXACT shape of every object with a random color.
        """
        if not self.ensure_data(): return
        if self.current_results.masks is None:
            messagebox.showinfo("Info", "No objects/shapes detected!")
            return

        temp = self.original_cv_image.copy()
        opacity = self.scale_color.get() / 100.0
        
        # Create a separate layer for the colored masks
        mask_overlay = temp.copy()
        random.seed(42)

        # Iterate over every detected mask
        # masks.xy is a list of coordinates for the outline of each object
        for contour in self.current_results.masks.xy:
            # Generate random color
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # Convert contour to integer numpy array
            cnt = np.array(contour, dtype=np.int32)
            
            # Fill the exact polygon shape on the overlay layer
            cv2.fillPoly(mask_overlay, [cnt], color)

        # Blend the overlay with the original image
        final_image = cv2.addWeighted(mask_overlay, opacity, temp, 1 - opacity, 0)
        
        self.show_image(final_image)
        self.lbl_status.config(text="Applied Random Color to Exact Shapes")

    def apply_mask_isolation(self):
        """
        Shows ONLY the shapes of the objects against a black background.
        """
        if not self.ensure_data(): return
        if self.current_results.masks is None: return

        # Start with black image
        black_bg = np.zeros_like(self.original_cv_image)
        
        for i, contour in enumerate(self.current_results.masks.xy):
            cnt = np.array(contour, dtype=np.int32)
            # Fill shape with white (or original object pixels)
            cv2.fillPoly(black_bg, [cnt], (255, 255, 255))
            
            # If you want the actual object pixels instead of white:
            # cv2.fillPoly(mask_layer, [cnt], 255)
            # res = cv2.bitwise_and(self.original_cv_image, self.original_cv_image, mask=mask_layer)
        
        self.show_image(black_bg)
        self.lbl_status.config(text="Shown Isolated Masks")

    def apply_contours(self):
        """
        Draws the outline (contours) of the exact shapes.
        """
        if not self.ensure_data(): return
        if self.current_results.masks is None: return

        temp = self.original_cv_image.copy()
        
        for contour in self.current_results.masks.xy:
            cnt = np.array(contour, dtype=np.int32)
            # Draw contours: image, contours, contourIdx, color, thickness
            cv2.drawContours(temp, [cnt], -1, (0, 255, 0), 2)
            
        self.show_image(temp)
        self.lbl_status.config(text="Feature Extraction: Contours Drawn")

    def apply_pattern_recognition(self):
        """Uses the built-in YOLO plotting which handles masking beautifully."""
        if not self.ensure_data(): return
        res_plot = self.current_results.plot(img=self.original_cv_image.copy())
        self.show_image(res_plot)
        self.lbl_status.config(text="Full Segmentation Recognition Displayed")

    # --- UTILS ---
    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.original_cv_image = cv2.imread(path)
            self.current_results = None # Reset results for new image
            self.show_image(self.original_cv_image)
            self.lbl_status.config(text="Image Loaded")

    def toggle_camera(self):
        if not self.is_camera_on:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_camera_on = True
                self.btn_camera.config(text="🛑 Stop Camera", bg="#c0392b")
                self.update_cam()
        else:
            self.is_camera_on = False
            if self.cap: self.cap.release()
            self.btn_camera.config(text="📷 Open Camera", bg="#2ecc71")

    def update_cam(self):
        if self.is_camera_on:
            ret, frame = self.cap.read()
            if ret:
                results = self.model(frame, conf=0.25, verbose=False)
                # Plotting segmentation on video frame
                annotated = results[0].plot()
                self.show_image(annotated)
                self.root.after(10, self.update_cam)

    def reset_image(self):
        if self.original_cv_image is not None:
            self.show_image(self.original_cv_image)
            self.lbl_status.config(text="Image Reset")

    def show_image(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        im.thumbnail((900, 700))
        tk_img = ImageTk.PhotoImage(image=im)
        self.lbl_image.configure(image=tk_img, text="")
        self.lbl_image.image = tk_img

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()