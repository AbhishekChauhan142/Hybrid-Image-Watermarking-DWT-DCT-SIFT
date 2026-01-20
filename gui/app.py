import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from .components import ImagePanel
from modules.watermarker import Watermarker
from modules.sift_engine import SIFTEngine
from modules.metrics import calculate_psnr, calculate_ssim, calculate_nc
from modules.attacks import apply_attack


class WatermarkApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Cyber-Security: Hybrid Watermarking System v2.0")
        self.geometry("1350x850")
        ctk.set_appearance_mode("Dark")

        # Logic Components
        # Inside WatermarkApp.__init__
        self.wm_logic = Watermarker(alpha=20.0)  # Higher alpha = More Robust
        self.sift_engine = SIFTEngine()

        # --- Data State ---
        self.host_img = None
        self.logo_img = None
        self.watermarked_img = None
        self.current_wm_image = None  # Tracks the latest (clean or attacked) image
        self.ref_kp, self.ref_des = None, None
        self.wm_shape = (32, 32)

        self._init_ui()

    def _init_ui(self):
        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color="#111111")
        self.sidebar.pack(side="left", fill="y")

        ctk.CTkLabel(self.sidebar, text="DASHBOARD", font=("Segoe UI", 24, "bold"), text_color="#4cc9f0").pack(pady=30)

        # SECTION: INPUTS
        ctk.CTkLabel(self.sidebar, text="1. DATA SOURCE", font=("Segoe UI", 12, "bold"), text_color="gray").pack(
            pady=(10, 5))
        ctk.CTkButton(self.sidebar, text="Load Host Image", fg_color="#3a0ca3", command=self.load_host).pack(pady=5,
                                                                                                             padx=20)
        ctk.CTkButton(self.sidebar, text="Load Logo", fg_color="#3a0ca3", command=self.load_logo).pack(pady=5, padx=20)

        # SECTION: ACTIONS
        ctk.CTkLabel(self.sidebar, text="2. PROTECTION", font=("Segoe UI", 12, "bold"), text_color="gray").pack(
            pady=(20, 5))
        ctk.CTkButton(self.sidebar, text="EMBED WATERMARK", fg_color="#2c6e49", hover_color="#1b432c", height=40,
                      font=("Segoe UI", 13, "bold"), command=self.embed).pack(pady=5, padx=20)
        ctk.CTkButton(self.sidebar, text="Save Watermarked", fg_color="#444444", command=self.save_result).pack(pady=5,
                                                                                                                padx=20)

        # SECTION: EXTRACTION (Highly Visible)
        ctk.CTkLabel(self.sidebar, text="3. VERIFICATION", font=("Segoe UI", 12, "bold"), text_color="gray").pack(
            pady=(20, 5))
        self.btn_extract = ctk.CTkButton(self.sidebar, text="EXTRACT NOW", fg_color="#f72585", hover_color="#b5179e",
                                         height=55, font=("Segoe UI", 16, "bold"), command=self.manual_extract)
        self.btn_extract.pack(pady=10, padx=20)

        # --- Main Content ---
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        # Top Row: Images
        self.row1 = ctk.CTkFrame(self.content, fg_color="transparent")
        self.row1.pack(fill="x", pady=10)

        self.p_host = ImagePanel(self.row1, "ORIGINAL HOST")
        self.p_host.pack(side="left", padx=10)

        self.p_logo_in = ImagePanel(self.row1, "INPUT LOGO", size=(200, 200))
        self.p_logo_in.pack(side="left", padx=10)

        self.p_wm_view = ImagePanel(self.row1, "WATERMARKED / ATTACKED VIEW")
        self.p_wm_view.pack(side="left", padx=10)

        # Bottom Row: Results & Analysis
        self.row2 = ctk.CTkFrame(self.content, fg_color="transparent")
        self.row2.pack(fill="both", expand=True)

        self.p_logo_out = ImagePanel(self.row2, "EXTRACTED LOGO RESULT", size=(200, 200))
        self.p_logo_out.pack(side="left", padx=10)

        # Control Panel Card
        self.analysis_card = ctk.CTkFrame(self.row2, fg_color="#1e1e1e", corner_radius=15, border_width=1,
                                          border_color="#333333")
        self.analysis_card.pack(side="right", fill="both", expand=True, padx=10)

        ctk.CTkLabel(self.analysis_card, text="ATTACK SIMULATION & METRICS", font=("Segoe UI", 16, "bold")).pack(
            pady=10)

        self.atk_menu = ctk.CTkOptionMenu(self.analysis_card, values=["Gaussian Noise", "Rotation", "JPEG Compression"],
                                          width=200)
        self.atk_menu.pack(pady=5)

        self.atk_slider = ctk.CTkSlider(self.analysis_card, from_=0, to=50, width=300)
        self.atk_slider.pack(pady=10)

        ctk.CTkButton(self.analysis_card, text="Simulate Cyber Attack", fg_color="#d90429",
                      command=self.run_attack_simulation).pack(pady=10)

        # Metrics Output
        self.m_frame = ctk.CTkFrame(self.analysis_card, fg_color="#252525", corner_radius=10)
        self.m_frame.pack(fill="x", padx=20, pady=15)

        self.lbl_psnr = ctk.CTkLabel(self.m_frame, text="PSNR: -- dB", font=("Consolas", 14))
        self.lbl_psnr.pack(side="left", expand=True, pady=10)
        self.lbl_ssim = ctk.CTkLabel(self.m_frame, text="SSIM: --", font=("Consolas", 14))
        self.lbl_ssim.pack(side="left", expand=True, pady=10)
        self.lbl_nc = ctk.CTkLabel(self.m_frame, text="NC: --", font=("Consolas", 18, "bold"), text_color="#4cc9f0")
        self.lbl_nc.pack(side="left", expand=True, pady=10)

    # --- Logic Operations ---

    def load_host(self):
        path = filedialog.askopenfilename()
        if path:
            self.host_img = cv2.imread(path, 0).astype(np.float32)
            self.p_host.update_image(self.host_img)

    def load_logo(self):
        path = filedialog.askopenfilename()
        if path:
            img = cv2.imread(path, 0)
            self.logo_img = cv2.resize(img, self.wm_shape)
            _, self.logo_img = cv2.threshold(self.logo_img, 127, 255, cv2.THRESH_BINARY)
            self.p_logo_in.update_image(self.logo_img)

    def embed(self):
        if self.host_img is None or self.logo_img is None:
            messagebox.showwarning("Incomplete", "Please load Host and Logo first.")
            return

        self.watermarked_img = self.wm_logic.embed(self.host_img, self.logo_img, key=123)
        self.current_wm_image = self.watermarked_img.copy()  # Set state to clean watermarked
        self.p_wm_view.update_image(self.watermarked_img)

        # Register SIFT for geometric robustness
        self.ref_kp, self.ref_des = self.sift_engine.get_features(self.watermarked_img)

        # Calculate Imperceptibility Metrics
        p = calculate_psnr(self.host_img, self.watermarked_img)
        s = calculate_ssim(self.host_img, self.watermarked_img)
        self.lbl_psnr.configure(text=f"PSNR: {p:.2f} dB")
        self.lbl_ssim.configure(text=f"SSIM: {s:.4f}")
        messagebox.showinfo("Success", "Watermark embedded and SIFT features registered.")

    def run_attack_simulation(self):
        if self.watermarked_img is None:
            messagebox.showerror("Error", "Embed a watermark first!")
            return

        atk_type = self.atk_menu.get()
        intensity = self.atk_slider.get()

        # Generate attacked version
        attacked = apply_attack(self.watermarked_img, atk_type, intensity)
        self.current_wm_image = attacked  # Update state to attacked image
        self.p_wm_view.update_image(attacked)

        messagebox.showinfo("Attack Simulated", f"Applied {atk_type}. Now click EXTRACT NOW.")

    def manual_extract(self):
        if self.current_wm_image is None:
            messagebox.showwarning("Warning", "Nothing to extract. Embed first!")
            return

        # 1. Geometry Correction (SIFT)
        corrected = self.sift_engine.correct_geometric_attacks(
            self.current_wm_image, self.ref_kp, self.ref_des
        )

        # 2. Hybrid Extraction
        extracted = self.wm_logic.extract(corrected, key=123, wm_shape=self.wm_shape)

        # 3. Update UI
        self.p_logo_out.update_image(extracted)

        # 4. FIX: Calculate NC properly
        nc_val = calculate_nc(self.logo_img, extracted)
        self.lbl_nc.configure(text=f"NC: {nc_val:.4f}")

        # Optional: update PSNR/SSIM for the current attacked view vs original host
        p = calculate_psnr(self.host_img, self.current_wm_image)
        s = calculate_ssim(self.host_img, self.current_wm_image)
        self.lbl_psnr.configure(text=f"PSNR: {p:.2f} dB")
        self.lbl_ssim.configure(text=f"SSIM: {s:.4f}")

    def save_result(self):
        if self.watermarked_img is None: return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            out = np.clip(self.watermarked_img, 0, 255).astype(np.uint8)
            cv2.imwrite(path, out)


if __name__ == "__main__":
    app = WatermarkApp()
    app.mainloop()