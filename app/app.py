import os
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
from package.utils import run_detection, save_new_labels

app = Flask(__name__, static_folder='static')

# Define project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

# Directory paths for labeling and detection uploads
LABELING_UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'datasets', 'labeling_uploads')
DETECTION_UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'datasets', 'detection_uploads')
TRAINING_FOLDER = os.path.join(PROJECT_ROOT, 'datasets', 'labeling_uploads', 'train')
LABELS_FOLDER = os.path.join(TRAINING_FOLDER, 'labels')
IMAGES_FOR_TRAINING = os.path.join(TRAINING_FOLDER, 'images')

# Create necessary folders if they don't exist
os.makedirs(LABELING_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRAINING_FOLDER, exist_ok=True)
os.makedirs(LABELS_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOR_TRAINING, exist_ok=True)

# Define paths to model weights and dataset configuration file
MODEL_PATH = os.path.join('models', 'best.pt')
DATASET_PATH = os.path.join(PROJECT_ROOT, 'static', 'data.yaml')

# Load YOLO and SAM models
yolo_model = YOLO(MODEL_PATH)
sam = sam_model_registry["vit_l"](checkpoint=os.path.join( "models", "sam_vit_l_0b3195.pth"))
sam_predictor = SamPredictor(sam)

# Class ID mapping
class_mapping = {"chips": 0, "void": 1}

@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload for detection
@app.route('/upload_for_detection', methods=['POST'])
def upload_for_detection():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    original_path = os.path.join(DETECTION_UPLOAD_FOLDER, filename)
    file.save(original_path)
    
    # Run YOLO + SAM detection and get detected image and void details
    detected_img, areas = run_detection(yolo_model, sam_predictor, original_path)
    
    # Save the processed result image
    result_filename = f'result_{filename}'
    result_path = os.path.join(DETECTION_UPLOAD_FOLDER, result_filename)
    cv2.imwrite(result_path, detected_img)
    
    # Pass the result image path and areas to the result template
    return render_template('result.html', result_image=result_filename, areas=areas, original_image=filename)

# Route to handle image upload for labeling
@app.route('/upload_for_labeling', methods=['POST'])
def upload_for_labeling():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    original_path = os.path.join(IMAGES_FOR_TRAINING, filename)
    file.save(original_path)
    
    return render_template('labeling.html', filename=filename)

# Route to handle label saving
@app.route('/label', methods=['POST'])
def save_labels():
    data = request.get_json()
    filename = data.get('file')
    labels = data.get('labels')
    image_width = data.get('image_width')
    image_height = data.get('image_height')

    # Prepare YOLO data
    yolo_data = []
    for label in labels:
        class_id = class_mapping.get(label["label"], -1)
        if class_id == -1:
            continue

        x_min = label["x_min"]
        y_min = label["y_min"]
        x_max = label["x_max"]
        y_max = label["y_max"]

        x_center = ((x_min + x_max) / 2) / image_width
        y_center = ((y_min + y_max) / 2) / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        # Append each label in YOLO format
        yolo_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Create or append to the label file corresponding to the image
    label_file_path = os.path.join(LABELS_FOLDER, f"{os.path.splitext(filename)[0]}.txt")
    
    # Check if the file exists; if it exists, append the new labels
    if os.path.exists(label_file_path):
        with open(label_file_path, 'a') as f:  # 'a' mode to append
            f.write("\n".join(yolo_data) + "\n")
    else:
        with open(label_file_path, 'w') as f:  # 'w' mode to create new file
            f.write("\n".join(yolo_data) + "\n")

    return jsonify({"status": "success", "message": "Labels saved successfully in YOLO format."})

# Define paths to model weights and dataset configuration file
MODEL_PATH = os.path.join('models', 'best.pt')  # Path to pretrained YOLO model weights
DATASET_PATH = os.path.join(PROJECT_ROOT,'datasets', 'data.yaml')  # Path to data.yaml configuration

@app.route('/start_training', methods=['POST'])
def start_training():
    try:
        # Initialize YOLO model with existing weights
        model = YOLO(MODEL_PATH)
        
        # Check if dataset configuration file exists
        if not os.path.exists(DATASET_PATH):
            return jsonify({"status": "error", "message": f"Dataset configuration file not found at {DATASET_PATH}"}), 400
        
        # Log paths for debugging
        print(f"Model path: {MODEL_PATH}")
        print(f"Dataset path: {DATASET_PATH}")
        
        # Start training with dataset configuration
        results = model.train(data=DATASET_PATH, epochs=10, imgsz=640)

        # Return success message after training
        return jsonify({"status": "success", "message": "YOLO model training completed!"})

    except Exception as e:
        # Return error message if training fails
        return jsonify({"status": "error", "message": str(e)}), 500


# Serve files from the 'datasets' folder
@app.route('/datasets/<path:filename>')
def serve_datasets(filename):
    return send_from_directory(os.path.join(PROJECT_ROOT, 'datasets'), filename)

if __name__ == '__main__':
    app.run(debug=True)