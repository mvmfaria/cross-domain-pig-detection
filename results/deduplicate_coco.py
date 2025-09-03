
import json
import os

def deduplicate_coco_annotations(input_path, output_path):
    """
    Deduplicates images and their corresponding annotations in a COCO-style annotation file.
    It assigns new sequential IDs to the unique images and updates the annotations accordingly.
    """
    with open(input_path, 'r') as f:
        coco_data = json.load(f)

    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    
    unique_images_by_filename = {}
    for image in images:
        # The 'id' for an image is sometimes stored under 'id' or 'image_id'
        image_id = image.get('id', image.get('image_id'))
        file_name = image.get('file_name')

        if file_name not in unique_images_by_filename:
            unique_images_by_filename[file_name] = image

    unique_images = list(unique_images_by_filename.values())
    
    # Create a mapping from old image IDs to new image IDs
    id_map = {img.get('id', img.get('image_id')): i + 1 for i, img in enumerate(unique_images)}
    
    # Update image IDs
    for i, image in enumerate(unique_images):
        old_id = image.get('id', image.get('image_id'))
        image['id'] = id_map[old_id]

    # Update annotation image_ids
    updated_annotations = []
    if annotations:
        for ann in annotations:
            old_image_id = ann.get('image_id')
            if old_image_id in id_map:
                ann['image_id'] = id_map[old_image_id]
                updated_annotations.append(ann)
    
    deduplicated_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'images': unique_images,
        'categories': coco_data.get('categories', []),
        'annotations': updated_annotations
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(deduplicated_data, f, indent=4)

    print(f"Deduplicated annotations saved to {output_path}")
    print(f"Original images: {len(images)}, Unique images: {len(unique_images)}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deduplicate COCO annotation file.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input COCO annotations file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output deduplicated COCO annotations file.')
    args = parser.parse_args()

    deduplicate_coco_annotations(args.input, args.output)
