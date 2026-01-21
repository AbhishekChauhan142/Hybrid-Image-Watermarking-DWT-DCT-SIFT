import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
from .components import ImagePanel
from modules.watermarker import Watermarker
from modules.sift_engine import SIFTEngine
from modules.metrics import calculate_psnr, calculate_ssim, calculate_nc
from modules.attacks import apply_attack


class WatermarkApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Cyber-Security: Hybrid Watermarking System v3.0")
        self.geometry("1350x850")
        ctk.set_appearance_mode("Dark")

        # Logic Components
        self.wm_logic = Watermarker()
        self.sift_engine = SIFTEngine()

        # --- Data State ---
        self.host_img = None
        self.logo_img = None
        self.watermarked_img = None
        self.current_wm_image = None
        self.ref_kp, self.ref_des = None, None
        self.wm_shape = (32, 32)

        self._init_ui()

    def _init_ui(self):
        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color="#111111")
        self.sidebar.pack(side="left", fill="y")

        ctk.CTkLabel(self.sidebar, text="DASHBOARD", font=("Segoe UI", 24, "bold"), text_color="#4cc9f0").pack(pady=30)

        # 1. DATA SOURCE
        ctk.CTkLabel(self.sidebar, text="1. DATA SOURCE", font=("Segoe UI", 12, "bold"), text_color="gray").pack(
            pady=(10, 5))
        ctk.CTkButton(self.sidebar, text="Load Host Image", fg_color="#3a0ca3", command=self.load_host).pack(pady=5,
                                                                                                             padx=20)
        ctk.CTkButton(self.sidebar, text="Load Logo", fg_color="#3a0ca3", command=self.load_logo).pack(pady=5, padx=20)
        ctk.CTkButton(self.sidebar, text="Load Watermarked", fg_color="#444444",
                      command=self.load_existing_watermark).pack(pady=5, padx=20)

        # 2. PARAMETERS (Security & Strength)
        ctk.CTkLabel(self.sidebar, text="2. PARAMETERS", font=("Segoe UI", 12, "bold"), text_color="gray").pack(
            pady=(20, 5))

        self.key_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Secret Key", justify="center")
        self.key_entry.pack(pady=5, padx=20)
        self.key_entry.insert(0, "123")

        # Alpha Slider with Dynamic Numbering
        ctk.CTkLabel(self.sidebar, text="Embedding Strength (Alpha)", font=("Segoe UI", 11), text_color="silver").pack(
            pady=(10, 0))
        self.lbl_alpha_val = ctk.CTkLabel(self.sidebar, text="Value: 20", font=("Consolas", 12, "bold"),
                                          text_color="#4cc9f0")
        self.lbl_alpha_val.pack()

        self.alpha_slider = ctk.CTkSlider(self.sidebar, from_=5, to=100, number_of_steps=19,
                                          command=self._update_alpha_label)
        self.alpha_slider.pack(pady=5, padx=20)
        self.alpha_slider.set(20)

        # 3. PROTECTION
        ctk.CTkLabel(self.sidebar, text="3. PROTECTION", font=("Segoe UI", 12, "bold"), text_color="gray").pack(
            pady=(20, 5))
        ctk.CTkButton(self.sidebar, text="EMBED WATERMARK", fg_color="#2c6e49", height=40,
                      font=("Segoe UI", 13, "bold"), command=self.embed).pack(pady=5, padx=20)
        ctk.CTkButton(self.sidebar, text="Save Result", fg_color="#444444", command=self.save_result).pack(pady=5,
                                                                                                           padx=20)

        # 4. VERIFICATION
        ctk.CTkLabel(self.sidebar, text="4. VERIFICATION", font=("Segoe UI", 12, "bold"), text_color="gray").pack(
            pady=(20, 5))
        self.btn_extract = ctk.CTkButton(self.sidebar, text="EXTRACT NOW", fg_color="#f72585", height=55,
                                         font=("Segoe UI", 16, "bold"), command=self.manual_extract)
        self.btn_extract.pack(pady=10, padx=20)

        # --- Main Content ---
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        # Row 1: Image Displays
        row1 = ctk.CTkFrame(self.content, fg_color="transparent")
        row1.pack(fill="x", pady=10)
        self.p_host = ImagePanel(row1, "ORIGINAL HOST")
        self.p_host.pack(side="left", padx=10)
        self.p_logo_in = ImagePanel(row1, "INPUT LOGO", size=(200, 200))
        self.p_logo_in.pack(side="left", padx=10)
        self.p_wm_view = ImagePanel(row1, "WATERMARKED / ATTACKED VIEW")
        self.p_wm_view.pack(side="left", padx=10)

        # Row 2: Results & Attack Simulation
        row2 = ctk.CTkFrame(self.content, fg_color="transparent")
        row2.pack(fill="both", expand=True)
        self.p_logo_out = ImagePanel(row2, "EXTRACTED LOGO RESULT", size=(200, 200))
        self.p_logo_out.pack(side="left", padx=10)

        self.analysis_card = ctk.CTkFrame(row2, fg_color="#1e1e1e", corner_radius=15, border_width=1,
                                          border_color="#333333")
        self.analysis_card.pack(side="right", fill="both", expand=True, padx=10)

        ctk.CTkLabel(self.analysis_card, text="ATTACK SIMULATION & METRICS", font=("Segoe UI", 16, "bold")).pack(
            pady=10)

        self.atk_menu = ctk.CTkOptionMenu(self.analysis_card, values=["Gaussian Noise", "Rotation", "JPEG Compression"])
        self.atk_menu.pack(pady=5)

        # Attack Intensity Slider with Dynamic Numbering
        self.lbl_atk_val = ctk.CTkLabel(self.analysis_card, text="Intensity: 25", font=("Consolas", 12, "bold"),
                                        text_color="#d90429")
        self.lbl_atk_val.pack(pady=(10, 0))

        self.atk_slider = ctk.CTkSlider(self.analysis_card, from_=0, to=100, command=self._update_atk_label)
        self.atk_slider.pack(pady=10)
        self.atk_slider.set(25)

        ctk.CTkButton(self.analysis_card, text="Simulate Cyber Attack", fg_color="#d90429",
                      command=self.run_attack_simulation).pack(pady=10)

        # Metrics Panel
        self.m_frame = ctk.CTkFrame(self.analysis_card, fg_color="#252525", corner_radius=10)
        self.m_frame.pack(fill="x", padx=20, pady=15)
        self.lbl_psnr = ctk.CTkLabel(self.m_frame, text="PSNR: --", font=("Consolas", 13))
        self.lbl_psnr.pack(side="left", expand=True, pady=10)
        self.lbl_ssim = ctk.CTkLabel(self.m_frame, text="SSIM: --", font=("Consolas", 13))
        self.lbl_ssim.pack(side="left", expand=True, pady=10)
        self.lbl_nc = ctk.CTkLabel(self.m_frame, text="NC: --", font=("Consolas", 18, "bold"), text_color="#4cc9f0")
        self.lbl_nc.pack(side="left", expand=True, pady=10)

    # --- Slider Value Callbacks ---
    def _update_alpha_label(self, value):
        self.lbl_alpha_val.configure(text=f"Value: {int(value)}")

    def _update_atk_label(self, value):
        self.lbl_atk_val.configure(text=f"Intensity: {int(value)}")

    # --- Logic Helpers ---
    def _update_metrics(self):
        if self.host_img is not None and self.current_wm_image is not None:
            p = calculate_psnr(self.host_img, self.current_wm_image)
            s = calculate_ssim(self.host_img, self.current_wm_image)
            self.lbl_psnr.configure(text=f"PSNR: {p:.2f} dB")
            self.lbl_ssim.configure(text=f"SSIM: {s:.4f}")
        else:
            self.lbl_psnr.configure(text="PSNR: No Host")
            self.lbl_ssim.configure(text="SSIM: No Host")

    def get_secret_key(self):
        try:
            return int(self.key_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Secret Key must be an integer!")
            return None

    # --- Button Commands ---
    def load_host(self):
        path = filedialog.askopenfilename()
        if path:
            img = cv2.imread(path, 0)
            h, w = img.shape
            new_h, new_w = (h if h % 2 == 0 else h - 1), (w if w % 2 == 0 else w - 1)
            self.host_img = cv2.resize(img, (new_w, new_h)).astype(np.float32)
            self.p_host.update_image(self.host_img)
            self.ref_kp, self.ref_des = self.sift_engine.get_features(self.host_img)
            self._update_metrics()

    def load_logo(self):
        path = filedialog.askopenfilename()
        if path:
            img = cv2.imread(path, 0)
            self.logo_img = cv2.resize(img, self.wm_shape)
            _, self.logo_img = cv2.threshold(self.logo_img, 127, 255, cv2.THRESH_BINARY)
            self.p_logo_in.update_image(self.logo_img)

    def load_existing_watermark(self):
        path = filedialog.askopenfilename()
        if path:
            img = cv2.imread(path, 0).astype(np.float32)
            self.watermarked_img = img
            self.current_wm_image = img.copy()
            self.p_wm_view.update_image(img)
            self._update_metrics()
            messagebox.showinfo("Loaded", "Ready to Extract. Use correct key.")

    def embed(self):
        key = self.get_secret_key()
        alpha = self.alpha_slider.get()
        if key is None or self.host_img is None or self.logo_img is None:
            messagebox.showwarning("Incomplete", "Load Host and Logo first.")
            return

        self.wm_logic.alpha = alpha
        self.watermarked_img = self.wm_logic.embed(self.host_img, self.logo_img, key)
        self.current_wm_image = self.watermarked_img.copy()
        self.p_wm_view.update_image(self.watermarked_img)
        self.ref_kp, self.ref_des = self.sift_engine.get_features(self.watermarked_img)
        self._update_metrics()
        self.lbl_nc.configure(text="NC: --")
        messagebox.showinfo("Success", f"Embedded (Strength: {int(alpha)})")

    def run_attack_simulation(self):
        if self.watermarked_img is None: return
        self.current_wm_image = apply_attack(self.watermarked_img, self.atk_menu.get(), self.atk_slider.get())
        self.p_wm_view.update_image(self.current_wm_image)

        # Auto-save Attack Result
        os.makedirs("data/output", exist_ok=True)
        cv2.imwrite("data/output/attacked_result.png", np.clip(self.current_wm_image, 0, 255).astype(np.uint8))
        self._update_metrics()
        self.lbl_nc.configure(text="NC: Pending")

    def manual_extract(self):
        key = self.get_secret_key()
        if key is None or self.current_wm_image is None: return

        if self.ref_kp is not None:
            corrected = self.sift_engine.correct_geometric_attacks(self.current_wm_image, self.ref_kp, self.ref_des)
        else:
            corrected = self.current_wm_image

        extracted = self.wm_logic.extract(corrected, key, self.wm_shape)
        self.p_logo_out.update_image(extracted)

        if self.logo_img is not None:
            nc_val = calculate_nc(self.logo_img, extracted)
            self.lbl_nc.configure(text=f"NC: {nc_val:.4f}", text_color="#4cc9f0")
        else:
            self.lbl_nc.configure(text="NC: No Ref Logo")
        self._update_metrics()

    def save_result(self):
        if self.current_wm_image is None: return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            cv2.imwrite(path, np.clip(self.current_wm_image, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    app = WatermarkApp()
    app.mainloop()