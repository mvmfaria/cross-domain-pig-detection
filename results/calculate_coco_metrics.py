import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def calculate_coco_metrics(ground_truth_path, predictions_path, output_path=None):
    """
    Calculates and prints COCO evaluation metrics.
    It handles cases where prediction image_ids are filenames instead of integer IDs.
    Optionally saves the metrics to a JSON file.
    """
    try:
        with open(ground_truth_path, 'r') as f:
            gt_data = json.load(f)
        
        filename_to_id = {img['file_name'].split('/')[-1]: img['id'] for img in gt_data['images']}

        with open(predictions_path) as f:
            preds_data = json.load(f)
        
        if not preds_data:
            return

        for pred in preds_data:
            if isinstance(pred['image_id'], str):
                filename = pred['image_id'].split('/')[-1]
                if filename in filename_to_id:
                    pred['image_id'] = filename_to_id[filename]
                else:
                    print(f"Warning: Filename {filename} from predictions not found in ground truth.")

    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading or parsing files: {e}")
        return
    except KeyError as e:
        print(f"KeyError: {e}. Check the structure of your JSON files.")
        return

    coco_gt = COCO(ground_truth_path)
    coco_dt = coco_gt.loadRes(preds_data)

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if output_path:
        stats = coco_eval.stats
        print(stats)
        metrics = {
            'AP_IoU_0.50_0.95_all_100': stats[0],
            'AP_IoU_0.50_all_100': stats[1],
            'AP_IoU_0.75_all_100': stats[2],
            'AP_IoU_0.50_0.95_small_100': stats[3],
            'AP_IoU_0.50_0.95_medium_100': stats[4],
            'AP_IoU_0.50_0.95_large_100': stats[5],
            'AR_IoU_0.50_0.95_all_1': stats[6],
            'AR_IoU_0.50_0.95_all_10': stats[7],
            'AR_IoU_0.50_0.95_all_100': stats[8],
            'AR_IoU_0.50_0.95_small_100': stats[9],
            'AR_IoU_0.50_0.95_medium_100': stats[10],
            'AR_IoU_0.50_0.95_large_100': stats[11]
        }
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {output_path}")