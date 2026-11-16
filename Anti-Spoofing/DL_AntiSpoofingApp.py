import cv2
import customtkinter as ctk
from ultralytics import YOLO
from PIL import Image, ImageTk
import sys
import os
import datetime # Added for logging timestamps
import time # To add a small delay

# --- Resource Path Function (For PyInstaller) ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# --- Application Class ---
class DL_AntiSpoofingApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Basic App Setup ---
        self.title("Face Anti-Spoofing System")
        self.geometry("1200x800") # Increased default size for better layout
        self.minsize(900, 700) # Increased min size
        
        # Set Global Scaling and Appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue") 
        ctk.set_widget_scaling(1.1)  # Increase overall widget size and font scaling
        ctk.set_window_scaling(1.0) # Keep window scaling to 1.0 (default)

        # --- Center the main window ---
        self.center_window()

        # --- Class Variables ---
        self.model = None
        self.cap = None
        self.current_status_text = "INITIALIZING..."
        self.status_color = "#36454F"  # Dark gray for initial status background
        self.status_text_color = "white" # Initial status text color
        self.confidence_threshold = 0.75 # Confidence interval for detection
        
        # This will be set by the new UI
        self.video_source = None 
        
        self.rotate_video = ctk.BooleanVar(value=False)


        # --- Configure Grid Layout for Main Window ---
        # Main window has 2 columns: [Sidebar] [Main Content Area]
        self.grid_columnconfigure(0, weight=0) # Sidebar (fixed width)
        self.grid_columnconfigure(1, weight=1) # Main content area (expands)
        self.grid_rowconfigure(0, weight=1) # Single row

        # --- 1. Sidebar Frame ---
        # --- FIX: Give the scrollable frame a fixed width ---
        self.sidebar_frame = ctk.CTkScrollableFrame(self, corner_radius=10, fg_color="#212121", width=300) # Added width=300
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # --- FIX: Configure column 0 of the sidebar to have weight=1 ---
        # This makes the child widgets fill the frame's width
        self.sidebar_frame.grid_columnconfigure(0, weight=1)

        # Sidebar Widgets
        self.sidebar_title_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Anti-Spoofing",
            font=ctk.CTkFont(size=24, weight="bold", slant="italic"),
            text_color="#BB86FC" # Purple accent
        )
        self.sidebar_title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        self.version_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="v1.4.1", # Updated version
            font=ctk.CTkFont(size=14),
            text_color="#A9A9A9" # Gray
        )
        self.version_label.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")

        # --- Camera Setup Section ---
        self.camera_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Camera Setup",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#E0E0E0"
        )
        self.camera_label.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.source_combobox = ctk.CTkComboBox(
            self.sidebar_frame,
            values=[
                "Default Webcam (0)",
                "DroidCam (1)",
                "DroidCam (2)",
                "http://192.168.1.10:8080/video"
            ],
            font=ctk.CTkFont(size=13)
        )
        self.source_combobox.grid(row=3, column=0, padx=20, pady=(5, 10), sticky="ew")

        self.connect_button = ctk.CTkButton(
            self.sidebar_frame,
            text="Start / Reconnect Camera",
            command=self.connect_to_source,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.connect_button.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="ew")
        
        self.rotate_checkbox = ctk.CTkCheckBox(
            self.sidebar_frame,
            text="Rotate 90° Counter-Clockwise",
            variable=self.rotate_video,
            font=ctk.CTkFont(size=13)
        )
        self.rotate_checkbox.grid(row=5, column=0, padx=20, pady=(0, 20), sticky="w")

        self.about_header_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="About",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#E0E0E0"
        )
        self.about_header_label.grid(row=6, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.about_text = ctk.CTkTextbox(
            self.sidebar_frame,
            height=200, # Increased height for new info
            # width=250, # REMOVED: Let it fill the frame
            wrap="word",
            font=ctk.CTkFont(size=13),
            text_color="#D3D3D3",
            fg_color="#2C2C2C", # Slightly lighter than sidebar background
            activate_scrollbars=False # Scrollbar is now on the frame
        )
        self.about_text.insert("0.0", 
            "This app uses a YOLOv11 model to detect real faces vs. spoofing attempts.\n\n"
            "--- Phone Camera (USB) ---\n"
            "For a fast, lag-free feed, use a USB cable:\n"
            "1. Install 'DroidCam' on your phone.\n"
            "2. Install 'DroidCam Client' on your PC.\n"
            "3. Enable 'USB Debugging' on your phone.\n"
            "4. Connect phone to PC via USB and start DroidCam.\n"
            "5. Select 'DroidCam (1)' or 'DroidCam (2)' from the dropdown above and click Start.\n\n"
            "--- Phone Camera (Wi-Fi) ---\n"
            "1. Install 'IP Webcam' on your phone.\n"
            "2. Type the URL (e.g., http://.../video) into the dropdown box.\n"
            "3. Click Start."
        )
        self.about_text.configure(state="disabled") # Make it read-only
        self.about_text.grid(row=7, column=0, padx=20, pady=10, sticky="ew")

        # --- Log Section ---
        self.log_header_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Detection Logs",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#E0E0E0"
        )
        self.log_header_label.grid(row=8, column=0, padx=20, pady=(10, 0), sticky="w")

        self.log_textbox = ctk.CTkTextbox(
            self.sidebar_frame,
            # width=250, # REMOVED: Let it fill the frame
            height=150, # Give it a default height
            wrap="word",
            font=ctk.CTkFont(size=12, family="Courier New"), # Monospaced font for logs
            text_color="#D3D3D3",
            fg_color="#2C2C2C"
        )
        self.log_textbox.grid(row=9, column=0, padx=20, pady=10, sticky="ew")
        self.log_textbox.insert("0.0", "--- System Log Initialized ---\n")
        self.log_textbox.configure(state="disabled") # Make it read-only
        
        # Status indicator within sidebar
        self.sidebar_status_frame = ctk.CTkFrame(self.sidebar_frame, height=40, corner_radius=8, fg_color=self.status_color)
        self.sidebar_status_frame.grid(row=11, column=0, padx=20, pady=(10, 20), sticky="ew")
        self.sidebar_status_frame.grid_columnconfigure(0, weight=1)
        self.sidebar_status_label = ctk.CTkLabel(
            self.sidebar_status_frame,
            text=f"Status: {self.current_status_text}",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.status_text_color
        )
        self.sidebar_status_label.grid(row=0, column=0, padx=10, pady=5)


        # --- 2. Main Content Frame ---
        self.main_content_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#363636") # Lighter main background
        self.main_content_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # --- UI LAYOUT FIX ---
        # Configure grid *before* adding widgets
        self.main_content_frame.grid_rowconfigure(0, weight=0) # Title
        self.main_content_frame.grid_rowconfigure(1, weight=1) # Video (This is the critical fix)
        self.main_content_frame.grid_rowconfigure(2, weight=0) # Main Status
        self.main_content_frame.grid_columnconfigure(0, weight=1)
        # --- END OF UI LAYOUT FIX ---


        # Main Content Widgets
        self.main_title_label = ctk.CTkLabel(
            self.main_content_frame,
            text="Live Video Feed",
            font=ctk.CTkFont(size=30, weight="bold"),
            text_color="#FFFFFF"
        )
        # --- UI LAYOUT FIX: Place title in row 0 ---
        self.main_title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="n")

        self.video_label = ctk.CTkLabel(
            self.main_content_frame,
            text="Waiting for video feed... \n\nSelect a camera source from the sidebar and click 'Start / Reconnect Camera'",
            font=ctk.CTkFont(size=20),
            fg_color="#1C1C1C", # Dark background for video area
            corner_radius=10
        )
        # --- UI LAYOUT FIX: Place video in row 1 ---
        self.video_label.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="nsew")
        
        # Main Status Bar (more prominent)
        self.main_status_frame = ctk.CTkFrame(self.main_content_frame, height=60, corner_radius=10, fg_color=self.status_color)
        # --- UI LAYOUT FIX: Place status in row 2 and make it fill horizontally ---
        self.main_status_frame.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.main_status_frame.grid_columnconfigure(0, weight=1)

        self.main_status_label = ctk.CTkLabel(
            self.main_status_frame,
            text=f"Current State: {self.current_status_text}",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=self.status_text_color
        )
        self.main_status_label.grid(row=0, column=0, padx=20, pady=10)

        # --- Start Application Logic ---
        self.load_model()
        self.update_frame() # Start the frame update loop (it will wait for self.cap)

    def center_window(self):
        """Centers the window on the screen."""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def add_log(self, message):
        """Adds a new log message to the top of the log textbox."""
        now = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{now}] {message}\n"
        
        self.log_textbox.configure(state="normal") # Enable writing
        self.log_textbox.insert("0.0", formatted_message) # Insert at the top
        
        # Limit log to 50 lines to prevent slowdown
        if int(self.log_textbox.index('end-1c').split('.')[0]) > 50:
            self.log_textbox.delete("50.0", "end")
            
        self.log_textbox.configure(state="disabled") # Disable writing

    def load_model(self):
        """Loads the YOLOv11 model using the resource_path."""
        try:
            self.add_log("Loading YOLOv11 model...")
            model_path = resource_path("best.pt")
            if not os.path.exists(model_path):
                self.update_status("Error: 'best.pt' not found.", "red", "#5a0000") # Dark red background
                self.add_log("Error: 'best.pt' not found.")
                return
            self.model = YOLO(model_path)
            self.update_status("Model Loaded. Select Camera.", "white", "#36454F") # Dark gray
            self.add_log("Model loaded successfully.")
        except Exception as e:
            self.update_status(f"Model Load Error: {e}", "red", "#5a0000")
            self.add_log(f"Model Load Error: {e}")

    def connect_to_source(self):
        """Gets the source from the combobox and starts the webcam."""
        source_text = self.source_combobox.get()
        
        if source_text == "Default Webcam (0)":
            self.video_source = 0
        elif source_text == "DroidCam (1)":
            self.video_source = 1
        elif source_text == "DroidCam (2)":
            self.video_source = 2
        else:
            self.video_source = source_text # Use the string URL
            
        self.add_log(f"Attempting to connect to: {self.video_source}")
        
        if self.cap is not None:
            self.cap.release()
            self.add_log("Released previous video source.")
            
        self.start_webcam()

    # --- Robustly try to open local webcams (like DroidCam) ---
    def _try_open_cam(self, index, backend):
        """Helper function to try opening a camera with a specific backend."""
        if backend is None:
            self.cap = cv2.VideoCapture(index)
            self.add_log(f"Trying backend: Default")
        else:
            self.cap = cv2.VideoCapture(index, backend)
            self.add_log(f"Trying backend: {backend}")

        # Nudge the driver by setting resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            self.add_log("...failed to open.")
            if self.cap: self.cap.release()
            return False

        # Warm-up loop
        self.add_log("...opened, warming up...")
        read_success = False
        for _ in range(10): # Try 10 times
            success, _ = self.cap.read()
            if success:
                self.add_log("...camera read successful.")
                read_success = True
                break
            time.sleep(0.1) # Wait 100ms
        
        if not read_success:
            self.add_log("...read failed.")
            if self.cap: self.cap.release()
            return False
            
        return True

    def start_webcam(self):
        """Initializes the webcam OR IP camera stream."""
        if self.video_source is None:
            self.add_log("No video source selected.")
            return

        try:
            self.add_log(f"Initializing video source: {self.video_source}")
            
            if isinstance(self.video_source, int):
                # It's a local camera (0, 1, 2...). Try all backends.
                self.add_log("Local camera detected. Trying DSHOW...")
                if self._try_open_cam(self.video_source, cv2.CAP_DSHOW):
                    pass # Success
                else:
                    self.add_log("DSHOW failed. Trying MSMF...")
                    if self._try_open_cam(self.video_source, cv2.CAP_MSMF):
                        pass # Success
                    else:
                        self.add_log("MSMF failed. Trying default backend...")
                        if self._try_open_cam(self.video_source, None):
                            pass # Success
                        else:
                            raise Exception("All backends failed to open camera.")

            else:
                # It's an IP stream. Use the default backend.
                self.cap = cv2.VideoCapture(self.video_source)
                if not self.cap.isOpened():
                    raise Exception("Capture object not opened for IP stream.")
                # Warm-up loop for IP stream
                self.add_log("Warming up IP camera...")
                success, _ = self.cap.read()
                if not success:
                    raise Exception("Failed to read frame from IP stream.")
            
            # If we get here, self.cap is valid and working
            self.add_log("Video source initialized successfully.")
            self.update_status("Video Connected. Detecting...", "white", "#36454F")

        except Exception as e:
            if self.cap:
                self.cap.release()
            self.cap = None
            self.update_status(f"Error: Cannot open video source.", "red", "#5a0000")
            self.add_log(f"Camera Error: {e}")
    # --- END OF FIX #3 ---

    def update_status(self, text, text_color, bg_color):
        """Updates the status labels text, text color, and background color."""
        if text != self.current_status_text:
            self.current_status_text = text
            self.status_text_color = text_color
            
            self.main_status_frame.configure(fg_color=bg_color)
            self.main_status_label.configure(text=f"Current State: {self.current_status_text}", text_color=self.status_text_color)
            
            self.sidebar_status_frame.configure(fg_color=bg_color)
            self.sidebar_status_label.configure(text=f"Status: {self.current_status_text}", text_color=self.status_text_color)

    def update_frame(self):
        """Main loop to read from webcam, run inference, and update UI."""
        
        if self.cap is None or not self.cap.isOpened():
            self.video_label.configure(text="Waiting for video feed... \n\nSelect a camera source from the sidebar and click 'Start / Reconnect Camera'", image=None)
            self.after(100, self.update_frame) # Keep checking for camera
            return
        
        success, frame = self.cap.read()
        if not success:
            self.update_status("Video feed ended.", "orange", "#8B4000") # Dark orange background
            self.add_log("Video feed ended or was interrupted.")
            self.cap.release()
            self.cap = None 
            self.after(100, self.update_frame) # Go back to waiting
            return
            
        # --- Rotate frame if checkbox is ticked ---
        if self.rotate_video.get():
            try:
                # Use 90 degrees counter-clockwise for portrait
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 
            except Exception as e:
                self.add_log(f"Error rotating frame: {e}")
                # Continue with unrotated frame
        # --- END OF NEW ---

        # --- Run Inference ---
        detected = False
        if self.model:
            results = self.model(frame, stream=True, conf=self.confidence_threshold, verbose=False) 
            
            for r in results:
                frame = r.plot() 
                
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    confidence = float(box.conf[0])
                    
                    if class_name == 'real':
                        self.update_status("REAL Face Detected", "#00FF00", "#004d00") 
                        if self.current_status_text != "REAL Face Detected":
                            self.add_log(f"REAL face detected. Conf: {confidence:.2f}")
                        detected = True
                    elif class_name == 'fake':
                        self.update_status("SPOOF Attack Detected!", "#FF0000", "#4d0000") 
                        if self.current_status_text != "SPOOF Attack Detected!":
                            self.add_log(f"SPOOF attack detected! Conf: {confidence:.2f}")
                        detected = True

            if not detected:
                if self.current_status_text not in ["REAL Face Detected", "SPOOF Attack Detected!"]:
                     self.update_status("No Face Detected", "white", "#36454F") 

        # --- Convert OpenCV Image (BGR) to CTkImage ---
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()

        if label_width > 10 and label_height > 10: 
            h, w, _ = frame.shape
            # Maintain aspect ratio
            if w / h > label_width / label_height:
                new_w = label_width
                new_h = int((h * label_width) / w)
            else:
                new_h = label_height
                new_w = int((w * label_height) / h)
                
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            ctk_image = ctk.CTkImage(light_image=pil_image, size=(new_w, new_h))
            
            self.video_label.configure(image=ctk_image, text="")
            self.video_label.image = ctk_image
        else:
            # This should no longer happen now that the layout is fixed
            self.video_label.configure(text="Resizing...", image=None)

        # 5. Repeat
        self.after(10, self.update_frame) # Run this function again after 10ms

    def on_closing(self):
        """Called when the window is closed."""
        if self.cap:
            self.cap.release()
            self.add_log("Video source released.")
        self.add_log("Application closing.")
        self.destroy()

if __name__ == "__main__":
    app = DL_AntiSpoofingApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
