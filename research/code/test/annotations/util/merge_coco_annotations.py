# pip install pycocotools
from pycocotools.coco import COCO
import json

"""
---------------------------------------------------------------------------------------
|  category   | #instances   |  category   | #instances   |  category  | #instances   |
|:-----------:|:-------------|:-----------:|:-------------|:----------:|:-------------|
|    Kumar    | 54           |    Esha     | 360          |  Shibangi  | 66           |
|   Anjali    | 198          | Bhalchandra | 108          |    Asha    | 60           |
|   Advait    | 24           |    Sham     | 18           |   Amanda   | 48           |
|  Nelakshi   | 48           |    sachi    | 48           |   sanvi    | 48           |
|    sagar    | 42           |   Jahnvi    | 6            |  Jawahar   | 6            |
| Chandrakant | 12           |    dipti    | 60           |   child    | 150          |
|     man     | 402          |    woman    | 690          |            |              |
|    total    | 2448         |             |              |            |              |

--------------------------------------------------------------------------------------
|  category  | #instances   |  category   | #instances   |  category  | #instances   |
|:----------:|:-------------|:-----------:|:-------------|:----------:|:-------------|
|   Kumar    | 81           |    Esha     | 540          |  Shibangi  | 99           |
|   Anjali   | 297          | Bhalchandra | 162          |    Asha    | 90           |
|   Advait   | 36           |    Sham     | 27           |   Amanda   | 72           |
|  Nelakshi  | 72           |    sachi    | 72           |   sanvi    | 72           |
|   sagar    | 63           | Chandrakant | 18           |   dipti    | 90           |
|   child    | 144          |     man     | 351          |   woman    | 909          |
|            |              |             |              |            |              |
|   total    | 3195         |             |              |            |              |



"""

def merge_coco_json(json_files, output_file):
    merged_annotations = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_id_offset = 0
    annotation_id_offset = 0
    category_id_offset = 0
    existing_category_ids = set()

    for idx, file in enumerate(json_files):
        coco = COCO(file)

        # Update image IDs to avoid conflicts
        for image in coco.dataset['images']:
            image['id'] += image_id_offset
            merged_annotations['images'].append(image)

        # Update annotation IDs to avoid conflicts
        for annotation in coco.dataset['annotations']:
            annotation['id'] += annotation_id_offset
            annotation['image_id'] += image_id_offset
            merged_annotations['annotations'].append(annotation)

        # Update categories and their IDs to avoid conflicts
        for category in coco.dataset['categories']:
            if category['id'] not in existing_category_ids:
                category['id'] += category_id_offset
                merged_annotations['categories'].append(category)
                existing_category_ids.add(category['id'])

        image_id_offset = len(merged_annotations['images'])
        annotation_id_offset = len(merged_annotations['annotations'])
        category_id_offset = len(merged_annotations['categories'])

    # Save merged annotations to output file
    with open(output_file, 'w') as f:
        json.dump(merged_annotations, f)

# List of paths to COCO JSON files to merge
json_files = ["../annotations.json", "../augmented_annotations.json"]

# Output file path for merged annotations
output_file = "../merged_annotations_coco.json"

# Merge COCO JSON files
merge_coco_json(json_files, output_file)

print("Merged COCO JSON files saved to", output_file)