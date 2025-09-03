import os
import sys
sys.path.append(os.path.dirname(__file__))
from calculate_coco_metrics import calculate_coco_metrics

def evaluate_all_models(results_dir, gt_file):
    """
    Avalia todos os predictions.json presentes nas subpastas de resultados dos modelos.
    Salva metrics.json em cada pasta.
    """
    for model_folder in os.listdir(results_dir):
        if not model_folder.endswith('_test'):
            continue
        model_path = os.path.join(results_dir, model_folder)
        if not os.path.isdir(model_path):
            continue
        predictions_path = os.path.join(model_path, "predictions.json")
        metrics_path = os.path.join(model_path, "metrics.json")
        if os.path.exists(predictions_path):
            calculate_coco_metrics(gt_file, predictions_path, metrics_path)
        else:
            continue

if __name__ == "__main__":
    evaluate_all_models("experiments", "data/PigLife/coco_format/test_deduplicated.json")