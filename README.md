Hybrid Robust Image Watermarking (DWT-DCT-SIFT)

A research-based implementation of a robust digital image watermarking system designed for copyright protection. This project implements a hybrid approach combining frequency-domain transforms (DWT and DCT) for signal robustness and computer vision descriptors (SIFT) for geometric invariance.
![alt text](https://img.shields.io/badge/python-3.10+-blue.svg)

![alt text](https://img.shields.io/badge/license-MIT-green.svg)

![alt text](https://img.shields.io/badge/Cyber_Security-Digital_Forensics-red.svg)

📖 Overview

Digital media can be easily duplicated and edited. This system embeds an invisible copyright logo into a host image. Unlike standard watermarking, this "Hybrid" method can recover the logo even if the image has been:
Attacked with Noise (Gaussian/Salt & Pepper)
Compressed (JPEG Compression)
Geometrically Distorted (Rotated or Resized)
How it Works:
Embedding: The host image is decomposed using DWT. The DCT is applied to the blocks of the sub-bands. A pseudo-random sequence representing the watermark bits is added to the mid-band DCT coefficients.
SIFT Registration: Scale-Invariant Feature Transform (SIFT) keypoints are extracted from the watermarked image and stored as a "Digital Signature."
Extraction & Correction: If an image is rotated, the system uses the stored SIFT signatures to re-align (correct) the image before extracting the watermark bits.

🚀 Key Features

Modern Dashboard UI: Built with CustomTkinter for a professional dark-mode experience.
Attack Simulator: Built-in tools to test robustness against Rotation, Noise, and JPEG Compression.
Scientific Metrics: Real-time calculation of:
PSNR (Peak Signal-to-Noise Ratio): Measures imperceptibility.
SSIM (Structural Similarity Index): Measures visual quality.
NC (Normalized Correlation): Measures the accuracy of logo recovery.


🛠️ Installation

Clone the repository:

git clone https://github.com/yourusername/HybridWatermarking.git
cd HybridWatermarking


Install dependencies:

pip install -r requirements.txt
Required libraries: opencv-python, PyWavelets, customtkinter, scikit-image, numpy, scipy.

Run the application:

python main.py

🖥️ Usage Guide

Load Host Image: Select a standard 512x512 grayscale image (e.g., Lena, Baboon).
Load Logo: Select a black-and-white copyright logo (will be resized to 32x32).
Embed Watermark: Click the green button. The system will hide the logo and generate SIFT descriptors.
Test Robustness (Optional):
Select an attack (e.g., Rotation) from the dropdown.
Adjust the intensity slider.
Click "Simulate Cyber Attack."
Extract Now: Click the pink button. The system re-aligns the image and recovers the logo, displaying the NC robustness score.
📊 Experimental Results
Attack Type	Typical NC Score	Status
No Attack	0.999+	✅ Perfect
Gaussian Noise	0.90 - 0.96	✅ Robust
JPEG (Quality 50)	0.98 - 0.99	✅ Robust
Rotation (up to 45°)	0.88 - 0.92	✅ Geometric Invariant

📂 Project Structure

HybridWatermarking/
├── main.py                 # App entry point
├── data/                   # Output storage
├── gui/                    
│   ├── app.py              # Main dashboard logic
│   └── components.py       # Styled UI Card components
└── modules/                
    ├── transforms.py       # DWT/DCT mathematical logic
    ├── watermarker.py      # Embedding/Extraction algorithms
    ├── sift_engine.py      # SIFT geometric correction
    ├── metrics.py          # PSNR, SSIM, NC calculations
    └── attacks.py          # Attack simulation logic

📜 References

This project is based on the methodology described in:
Hamidi, M., et al. "A Hybrid Robust Image Watermarking Method Based on DWT-DCT and SIFT for Copyright Protection." J. Imaging 2021, 7, 218.