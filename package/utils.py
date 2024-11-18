import json
import cv2
import numpy as np
from loguru import logger

LABELS_FILE_PATH = 'retrain_data.txt'  # Central file for YOLO-format labels

def save_new_labels(label_info, image_width, image_height):
    """
    Saves labels in YOLO format for multiple images in a single file.
    """
    label_data = json.loads(label_info)  # Deserialize the JSON string

    # Open the central labels file in append mode
    with open(LABELS_FILE_PATH, 'a') as f:
        for label in label_data:
            class_id = label["class"]
            x_min = label["x_min"]
            y_min = label["y_min"]
            x_max = label["x_max"]
            y_max = label["y_max"]

            # Convert to YOLO format
            x_center = ((x_min + x_max) / 2) / image_width
            y_center = ((y_min + y_max) / 2) / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height

            # Write the label in YOLO format
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def run_detection(yolo_model, sam_predictor, image_path):
    """
    Runs YOLO and SAM on the given image, applying detection and segmentation.
    """
    # Load the image
    img = cv2.imread(image_path)
    logger.debug(f"Loaded image with shape: {img.shape}")
    # Run YOLO inference
    logger.debug("Running YOLO inference...")
    results = yolo_model(img)
    logger.debug(f"YOLO detected {len(results[0].boxes)} objects")
    
    voids = []
    for idx, (box, confidence, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
        xmin, ymin, xmax, ymax = map(int, box.tolist())
        area = (xmax - xmin) * (ymax - ymin)  # Calculate area
        
        # Prepare data for voids list
        void_info = {
            "id": idx + 1,
            "area": area,
            "unit": "pixels²",
            "coordinates": [xmin, ymin, xmax, ymax]
        }
        voids.append(void_info)
        
        logger.debug("we start the sam predictor")
        # Apply SAM for precise mask
        sam_predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        logger.debug("we set the image")
        masks, _, _ = sam_predictor.predict(
            point_coords=np.array([[xmin, ymin]], dtype=np.float32),
            point_labels=np.array([1]),
            box=np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        )
        
        logger.debug(f"Got {len(masks)} masks")
        if masks is not None and len(masks) > 0:
            mask = masks[0].astype(bool)
            img[mask] = (0, 255, 0)  # Highlight mask in green
        
    return img, voids








