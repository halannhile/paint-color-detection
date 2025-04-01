# Color Calibration System

A computer vision system for accurate color reproduction across varying lighting conditions and camera settings.

## Project Overview

In interior design, digital media production, and many other fields, accurately identifying colors from photographs is challenging due to varying lighting conditions and camera settings. This project addresses this challenge by developing a system that uses a physical color calibration tool to extract true color values from images.

The system consists of:
1. A physical calibration tool with red circle, green triangle, and blue pentagon shapes
2. An object detection model (YOLOv8) to identify these shapes and the target color
3. Multiple prediction models to calculate true RGB values from observed colors
4. A web application for easy use

## Project Report 

View our detailed project report [here](https://github.com/halannhile/paint-color-detection/blob/master/cosi149b_project1_report.pdf).

## Repository Structure

```
├── app.py                                              # Web application for color prediction
├── eval.py                                             # Evaluation metrics for color prediction models
├── main.py                                             # Entry point for classical ML models
├── nn_tree_ensemble.py                                 # Neural network and tree-based ensemble model
├── Color_Detection_Final_Model_Gemini_English.ipynb    # Autoencoder model
├── utils.py                                            # Utility functions for data processing and model handling
├── data/                                               # Dataset directory
│   ├── train.csv                                       # Training data
│   ├── val.csv                                         # Validation data
│   └── test.csv                                        # Test data
├── static/                                             # Static files for web application
└── templates/                                          # HTML templates for web application
```

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

The requirements include:
- PyTorch
- OpenCV
- Flask
- Ultralytics (YOLOv8)
- Scikit-learn
- XGBoost
- LightGBM
- Pandas
- NumPy
- Matplotlib

### Physical Calibration Tool

To use this system, you'll need to print the color calibration tool provided in the `calibration_tool.pdf` file (to be added soon!). This tool contains three reference shapes with known colors:

- Red circle (RGB: 255, 0, 0)
- Green triangle (RGB: 0, 255, 0)
- Blue pentagon (RGB: 0, 0, 255)

## Running the Code

### Training and Evaluating Classical ML Models

```bash
python main.py
```

This script will train and evaluate the following models:
- Linear Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- LightGBM
- AdaBoost

### Training and Evaluating the Neural Network Ensemble

```bash
python nn_tree_ensemble.py
```

This script trains the neural network and tree-based ensemble model, which combines:
- A custom neural network with attention mechanisms
- Gradient Boosting Regressor
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor

### Running the Web Application

```bash
python app.py
```

This will start a local web server at `http://127.0.0.1:5000/` where you can:
1. Upload an image containing your color calibration tool and the target color
2. View the detected objects and extracted colors
3. Select a prediction model
4. Get the predicted true RGB values

## Model Performance

Our models achieve the following performance metrics on the test dataset:

| Model | R² Score | RMSE | MAPE (%) | Mean ΔE | Median ΔE |
|-------|----------|------|----------|---------|-----------|
| Linear Regression | 0.051 | 0.1265 | 18.97 | 0.0406 | 0.0323 |
| Random Forest | 0.243 | 0.1136 | 9.53 | 0.0195 | 0.0130 |
| SVM | 0.265 | 0.1132 | 10.56 | 0.0225 | 0.0173 |
| XGBoost | 0.196 | 0.1173 | 10.54 | 0.0296 | 0.0111 |
| LightGBM | 0.268 | 0.1129 | 8.89 | 0.0201 | 0.0118 |
| AdaBoost | 0.279 | 0.1112 | 9.82 | 0.0215 | 0.0141 |
| Autoencoder | 0.920 | 0.0372 | 4.05 | 0.0244 | 0.0174 |
| NN-Tree Ensemble | 0.924 | 0.0355 | 3.61 | 0.0168 | 0.0111 |

The Neural Network and Tree-based Ensemble model achieves the best overall performance.

## Usage Guide

### Taking Photos for Color Calibration

For best results:
1. Place the calibration tool adjacent to the target color
2. Ensure all shapes are clearly visible and not shadowed
3. Take the photo in the lighting conditions you want to calibrate for
4. Avoid extreme angles that distort the shapes

### Using the Web Application

1. Upload your image using the "Choose File" button
2. Click "Upload & Extract Colors"
3. Select your preferred prediction model from the dropdown
4. Click "Predict True Color" to get the calibrated RGB values
5. The predicted color will be displayed as a swatch along with its RGB values

## Contributors

- Zepeng Hu
- Nhi Le
- Yurim Lee


## Acknowledgments

This project was developed for COSI 149B Practical Machine Learning with Big Data (Brandeis University) under the guidance of Prof. Pengyu Hong.
