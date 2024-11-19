# final-project-deployment

Quality control is of vital importance during electronics production. As the methods of producing electronic circuits improve, there is an increasing chance of solder defects during assembling the printed circuit board (PCB). Technology like X-ray imaging is used for inspection.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Images](#images)


## Introduction

This application is a web-based tool designed for labeling images and training a YOLO (You Only Look Once) model for object detection. It is primarily used for quality control in electronics production, specifically for detecting defects in printed circuit boards (PCBs) using X-ray imaging.

## Features

1. **Image Upload for Detection**:
   - Users can upload images for defect detection.
   - The application uses a pre-trained YOLO model and SAM (Segment Anything Model) to detect and highlight defects in the uploaded images.
   - Detected results are displayed along with the original image.

2. **Image Upload for Labeling**:
   - Users can upload images for manual labeling.
   - A canvas is provided where users can draw bounding boxes around detected areas and label them as either "chips" or "void".
   - The labeled data is saved in YOLO format for training purposes.

3. **Training the YOLO Model**:
   - Users can start training the YOLO model using the labeled data.
   - The application manages the training process and updates the model with new weights.
   - Training results, including images and performance metrics, are displayed to the user.

4. **Model Management**:
   - The application ensures that the necessary YOLO and SAM models are downloaded and available.
   - It manages the storage and retrieval of model weights and training data.

## Architecture

The application follows a modular architecture with the following components:

1. **User Interface**:
   - HTML Templates: [app/templates/index.html](app/templates/index.html), [app/templates/labeling.html](app/templates/labeling.html), [app/templates/result.html](app/templates/result.html)
   - JavaScript Code: Handles user interactions, canvas drawing, and AJAX requests.
   - CSS Styling: Provides styling for the web pages.

2. **Flask Backend**:
   - Routes & Views: Defined in [app/app.py](app/app.py) to handle different endpoints like image upload, label saving, and training initiation.
   - Business Logic: Processes user inputs, manages file uploads, and coordinates between different components.
   - Data Processing: Converts labeled data to YOLO format and prepares data for training.

3. **Model Management**:
   - YOLO & SAM Models: Managed in [package/utils.py](package/utils.py) and loaded in [app/app.py](app/app.py).
   - Model Training: Initiates and monitors the training process, updates model weights.
   - Model Inference: Runs detection on uploaded images using the YOLO model.

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/final-project-deployment.git
   cd final-project-deployment

2. **Build the Docker image**:
   ```sh
   docker build -t final-project-deployment .

3. **Run the Docker container**:
   ```sh
   docker run -p 8000:8000 final-project-deployment

## Usage

1. **Access the application**:
   Open your web browser and navigate to `http://localhost:8000`.

2. **Upload an image for detection**:
   - Go to the "Upload for Detection" section.
   - Choose an image and click "Detect".
   - View the detection results.

3. **Upload an image for labeling**:
   - Go to the "Upload for Labeling" section.
   - Choose an image and click "Label".
   - Draw bounding boxes and label the detected areas.
   - Save the labels.

4. **Start training the YOLO model**:
   - Click the "Start Training" button.
   - Monitor the training progress and view the results.

## Project Images

### Home
![Home](./project_images/Screenshot_2023-11-19_at_01.49.38.png)

### Labeling Interface
![Labeling Interface](./project_images/Screenshot_2023-11-19_at_01.53.34.png)

### Detection Results
![Detection Results 1](./project_images/Screenshot_2023-11-19_at_01.59.04.png)

![Detection Results 2](./project_images/Screenshot_2023-11-19_at_01.59.10.png)
