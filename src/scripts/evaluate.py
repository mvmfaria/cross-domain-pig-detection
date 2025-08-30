import os
from ultralytics import YOLO

EXPERIMENTS_PATH = "experiments"
YAML_FILE = "src/scripts/piglife.yaml"

models = os.listdir(EXPERIMENTS_PATH)

for model_name in models:

    model_path = os.path.join(EXPERIMENTS_PATH, model_name, "weights", "best.pt")
    model = YOLO(model_path)

    model.val(
        data=YAML_FILE,
        split="test",
        project=EXPERIMENTS_PATH,
        name=f"{model_name}_test",
        save_json=True
    )

    del model