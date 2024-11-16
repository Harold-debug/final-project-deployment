import os
import requests
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
from package.utils import run_detection, save_new_labels
import shutil

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

# Define model URLs and paths
YOLO_MODEL_URL = "https://github.com/Harold-debug/final-project-deployment/releases/download/yolo-model/best.pt"
SAM_MODEL_URL = "https://github.com/Harold-debug/final-project-deployment/releases/download/sam-checkpointer/sam_vit_l_0b3195.pth"
YOLO_MODEL_PATH = os.path.join('models', 'best.pt')
SAM_MODEL_PATH = os.path.join('models', 'sam_vit_l_0b3195.pth')

# Function to download files from GitHub
def download_file(url, destination):
    print(f"Downloading model from {url} to {destination}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded successfully: {destination}")
    else:
        print(f"Failed to download {url}: {response.status_code}")

# Check if models exist, if not download them
def ensure_models_downloaded():
    if not os.path.exists(YOLO_MODEL_PATH):
        download_file(YOLO_MODEL_URL, YOLO_MODEL_PATH)
    
    if not os.path.exists(SAM_MODEL_PATH):
        download_file(SAM_MODEL_URL, SAM_MODEL_PATH)

# Ensure models are downloaded when app starts
ensure_models_downloaded()

# Load YOLO and SAM models
yolo_model = YOLO(YOLO_MODEL_PATH)
sam = sam_model_registry["vit_l"](checkpoint=SAM_MODEL_PATH)
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

        # Find the most recent training directory by sorting based on creation time
        runs_detect_path = os.path.join(PROJECT_ROOT, 'runs', 'detect')
        
        # Get all subdirectories within 'runs/detect'
        training_folders = [f for f in os.listdir(runs_detect_path) if os.path.isdir(os.path.join(runs_detect_path, f))]

        # Find the most recent folder by creation time (newest first)
        most_recent_folder = max(training_folders, key=lambda folder: os.path.getctime(os.path.join(runs_detect_path, folder)))
        
        # Path to the most recent training folder
        most_recent_path = os.path.join(runs_detect_path, most_recent_folder)
        
        # Path to the best.pt model in the latest training folder
        new_best_model_path = os.path.join(most_recent_path, 'weights', 'best.pt')
        
        # Check if 'best.pt' exists and copy it to the model directory
        if os.path.exists(new_best_model_path):
            shutil.copy(new_best_model_path, MODEL_PATH)  # Copy new model to 'models' folder
            print(f"New best model saved to {MODEL_PATH}")
        
        # Collect image paths to display from the most recent folder
        images_to_display = [
            'results.png',
            'F1_curve.png',
            'train_batch0.jpg',
            'train_batch1.jpg'
        ]
        
       # Convert image file names to their relative paths, assuming they're in the training folder (not in weights)
        image_paths = [os.path.relpath(os.path.join(most_recent_path, img), runs_detect_path) 
                       for img in images_to_display if os.path.exists(os.path.join(most_recent_path, img))]
        print("Image Paths:", image_paths)
        # Return success message and image paths for frontend
        return jsonify({
            "status": "success",
            "message": "YOLO model training completed!",
            "images": images_to_display
        })

    except Exception as e:
        # Return error message if training fails
        return jsonify({"status": "error", "message": str(e)}), 500


# Serve files from the 'datasets' folder
@app.route('/datasets/<path:filename>')
def serve_datasets(filename):
    return send_from_directory(os.path.join(PROJECT_ROOT, 'datasets'), filename)

# Define the path to the 'runs/detect' folder
RUNS_DETECT_PATH = os.path.join(PROJECT_ROOT, 'runs', 'detect')

# Serve all files from the 'runs/detect' folder
@app.route('/runs/<path:filename>')
def serve_runs(filename):
    try:
        # Get the most recent training folder within 'runs/detect'
        training_folders = [f for f in os.listdir(RUNS_DETECT_PATH) if os.path.isdir(os.path.join(RUNS_DETECT_PATH, f))]
        if not training_folders:
            return "No training folders found", 404

        most_recent_folder = max(training_folders, key=lambda folder: os.path.getctime(os.path.join(RUNS_DETECT_PATH, folder)))

        # Define the path to the most recent folder
        most_recent_path = os.path.join(RUNS_DETECT_PATH, most_recent_folder)

        # Log for debugging: print paths
        print(f"Serving files from: {most_recent_path}")
        print(f"Requested file: {filename}")

        # Ensure the file exists in the directory
        file_path = os.path.join(most_recent_path, filename)
        if os.path.exists(file_path):
            print(f"File exists: {file_path}")  # Print the confirmed file path
            return send_from_directory(most_recent_path, filename)
        else:
            print(f"File not found: {file_path}")  # Log the error if file doesn't exist
            return f"File not found: {filename}", 404

    except Exception as e:
        # Log and return error if something goes wrong
        print(f"Error serving file: {str(e)}")
        return "Error serving file", 500
    
if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)