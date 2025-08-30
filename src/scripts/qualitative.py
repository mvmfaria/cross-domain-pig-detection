from ultralytics import YOLO
import os

model = YOLO("experiments/yolov8m/weights/best.pt")

images = [os.path.join("src/scripts/", f) for f in os.listdir("src/scripts/") if f.lower().endswith('.jpg')]

results = model(images)

for img_path, result in zip(images, results):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path = f"src/scripts/qualitative_{img_name}.jpg"
    result.save(filename=save_path, line_width=3, font_size=16)