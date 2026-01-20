from gui.app import WatermarkApp # Not WatermarkingApp
import os

def init_folders():
    paths = ['data/original', 'data/watermark', 'data/output']
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)

if __name__ == "__main__":
    init_folders()
    app = WatermarkApp()
    app.mainloop()