from ultralytics import YOLO
import os
import gc

PROJECT_NAME = "experiments"
DATA_CONFIG = "src/scripts/piglife.yaml"
EPOCHS = 100
IMGSZ = 640
BATCH = 4
WORKERS = 2

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
models_dir = os.path.join(project_dir, "src", "models")
model_files = os.listdir(models_dir)

for model_filename in model_files:
    model_path = os.path.join(models_dir, model_filename)
    run_name = os.path.splitext(model_filename)[0]
    
    model = YOLO(model_path)
    model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        project=PROJECT_NAME,
        name=run_name
    )

    del model

    best_weights_path = os.path.join(project_dir, PROJECT_NAME, run_name, 'weights', 'best.pt')
    
    val_model = YOLO(best_weights_path)
    val_model.val(
        data=DATA_CONFIG,
        split="test",
        project=PROJECT_NAME,
        name=f"{run_name}_val",
        save_json=True
    )

    del val_model
    gc.collect()