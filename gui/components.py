import customtkinter as ctk
from PIL import Image
import numpy as np


class ImagePanel(ctk.CTkFrame):
    def __init__(self, master, label_text, size=(280, 280), **kwargs):
        super().__init__(master, fg_color="#1e1e1e", border_width=2, border_color="#333333", corner_radius=15, **kwargs)
        self.size = size

        # Card Header
        self.header = ctk.CTkLabel(self, text=label_text, font=("Segoe UI", 12, "bold"),
                                   fg_color="#333333", corner_radius=5)
        self.header.pack(fill="x", padx=10, pady=(10, 5))

        # Image Area
        self.display_label = ctk.CTkLabel(self, text="No Image", width=size[0], height=size[1], text_color="#555555")
        self.display_label.pack(expand=True, fill="both", padx=15, pady=15)

    def update_image(self, cv_img):
        if cv_img is None: return
        # Pre-process numpy array for PIL
        if cv_img.dtype != np.uint8:
            cv_img = np.clip(cv_img, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(cv_img)
        # Fix Scaling Warning using CTkImage
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=self.size)

        self.display_label.configure(image=ctk_img, text="")
        self.display_label._image_ref = ctk_img  # Garbage collection safety