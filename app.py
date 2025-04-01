import os
import shutil
import numpy as np
import torch
import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
from utils import get_model
from eval import evaluate_metrics
import torch.nn as nn
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)
os.makedirs('temp/predict', exist_ok=True)
os.makedirs('temp/shapes', exist_ok=True)
os.makedirs('temp/centers', exist_ok=True)

# load all models at once during startup
try:
    yolo_model = YOLO("./trained_yolo8.pt")
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    yolo_model = None

# Neural Network model definition for loading saved models
class PerceptualColorNN(nn.Module):
    def __init__(self, input_dim, hidden_size=512):
        super(PerceptualColorNN, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        self.res_blocks = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15)
        )
        
        self.output_layer = nn.Linear(hidden_size, 3)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_rgb_codes(image_path):
    """Extract RGB values from an image using YOLO model"""
    print(f"Extracting RGB values from: {image_path}")
    
    # clear temporary directories
    for folder in ['temp/predict', 'temp/shapes', 'temp/centers']:
        os.makedirs(folder, exist_ok=True)
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    # run YOLO prediction
    if yolo_model is None:
        raise Exception("YOLO model not loaded")
        
    results = yolo_model.predict(
        source=image_path,
        save=True,
        conf=0.5,
        project="temp",
        name="predict"
    )
    
    print(f"YOLO detected {len(results[0].boxes)} objects")
    
    # original image
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Could not read image: {image_path}")
    
    # create a copy for drawing the detection results
    detection_image = image.copy()
    
    # extract detected objects
    extracted_colors = {}
    square_size = 20  # size of center square for color sampling
    
    # process each detected object and print details
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box coordinates
        class_id = int(box.cls[0])  # class index
        class_name = yolo_model.names[class_id]  # class label
        
        print(f"Detected {class_name} at [{x1}, {y1}, {x2}, {y2}]")
        
        # draw bounding box on detection image
        cv2.rectangle(detection_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(detection_image, class_name, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # crop the detected object
        cropped_object = image[y1:y2, x1:x2]
        shape_path = os.path.join('temp/shapes', f"{class_name}_{i}.jpg")
        cv2.imwrite(shape_path, cropped_object)
        
        # extract center color
        h, w, _ = cropped_object.shape
        center_x, center_y = w // 2, h // 2
        
        x1_center, y1_center = max(center_x - square_size // 2, 0), max(center_y - square_size // 2, 0)
        x2_center, y2_center = min(center_x + square_size // 2, w), min(center_y + square_size // 2, h)
        
        center_square = cropped_object[y1_center:y2_center, x1_center:x2_center]
        center_path = os.path.join('temp/centers', f"center_{class_name}_{i}.jpg")
        cv2.imwrite(center_path, center_square)
        
        # calculate average RGB (convert from BGR)
        avg_color = np.mean(center_square, axis=(0, 1))[::-1]  # BGR to RGB
        extracted_colors[class_name] = avg_color
        print(f"{class_name} avg RGB: {avg_color}")
    
    # save manually created detection image
    result_filename = os.path.basename(image_path)
    detection_image_path = os.path.join('static', 'results', f"result_{result_filename}")
    cv2.imwrite(detection_image_path, detection_image)
    
    # organize RGB values in the expected order with better fallback values
    rgb_list = []
    
    # check for paint color (looking for 'paint-color' in class name)
    paint_key = next((k for k in extracted_colors.keys() if 'paint-color' in k.lower()), None)
    if paint_key:
        rgb_list.extend(extracted_colors[paint_key])
        print(f"Found paint color: {extracted_colors[paint_key]}")
    else:
        print("WARNING: No paint-color detected, using default")
        rgb_list.extend([128, 128, 128])  # use gray as default
    
    # check for red circle
    circle_key = next((k for k in extracted_colors.keys() if 'circle' in k.lower()), None)
    if circle_key:
        rgb_list.extend(extracted_colors[circle_key])
        print(f"Found circle: {extracted_colors[circle_key]}")
    else:
        print("WARNING: No circle detected, using default")
        rgb_list.extend([255, 0, 0])  # default red
    
    # check for green triangle
    triangle_key = next((k for k in extracted_colors.keys() if 'triangle' in k.lower()), None)
    if triangle_key:
        rgb_list.extend(extracted_colors[triangle_key])
        print(f"Found triangle: {extracted_colors[triangle_key]}")
    else:
        print("WARNING: No triangle detected, using default")
        rgb_list.extend([0, 255, 0])  # default green
    
    # check for blue pentagon
    pentagon_key = next((k for k in extracted_colors.keys() if 'pentagon' in k.lower()), None)
    if pentagon_key:
        rgb_list.extend(extracted_colors[pentagon_key])
        print(f"Found pentagon: {extracted_colors[pentagon_key]}")
    else:
        print("WARNING: No pentagon detected, using default")
        rgb_list.extend([0, 0, 255])  # default blue
    
    print(f"Complete RGB list: {rgb_list}")
    return rgb_list, extracted_colors, result_filename

def visualize_detected_objects(save_path):
    """Display detected objects and save to static file"""
    if not os.path.exists('temp/shapes'):
        return None
        
    cropped_images = [os.path.join('temp/shapes', img) for img in os.listdir('temp/shapes') if img.endswith(".jpg")]
    
    if not cropped_images:
        return None

    fig, axes = plt.subplots(1, len(cropped_images), figsize=(15, 5))
    
    # handle case with single image
    if len(cropped_images) == 1:
        axes = [axes]
        
    for ax, img_path in zip(axes, cropped_images):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            ax.imshow(img)
            ax.set_title(os.path.basename(img_path))
            ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return save_path

def visualize_extracted_colors(extracted_colors, save_path):
    """Visualize extracted colors and save to static file"""
    if not extracted_colors:
        return None

    # filter to include only the desired classes
    desired_classes = ['circle', 'pentagon', 'triangle', 'paint-color']
    filtered_colors = {}
    
    for name, color in extracted_colors.items():
        # check if the class name contains any of the desired classes
        for cls in desired_classes:
            if cls.lower() in name.lower():
                filtered_colors[name] = color
                break
    
    if not filtered_colors:
        # if no desired classes found, use the original colors
        filtered_colors = extracted_colors
    
    # convert RGB values to 0-1 scale for Matplotlib
    normalized_colors = {name: np.array(color) / 255 for name, color in filtered_colors.items()}
    
    fig, axes = plt.subplots(1, len(normalized_colors), figsize=(10, 3))
    
    # handle case with single color
    if len(normalized_colors) == 1:
        axes = [axes]
        
    for i, (name, color) in enumerate(normalized_colors.items()):
        axes[i].imshow([[color]])
        axes[i].set_title(name, fontsize=10)
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return save_path

def predict_color(rgb_list, model_name):
    """Use selected model to predict true color from RGB values"""
    x = np.array(rgb_list, dtype=np.float32).reshape(1, -1)    
    print(f"Input shape: {x.shape}, values: {x}")
    
    # normalize values to 0-1 range for model input
    x_normalized = x / 255.0
    
    if model_name == 'neural_network':
        model_path = './results/perceptual_model/perceptual_model.pth'
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                input_dim = checkpoint.get('input_dim', 12)
                model = PerceptualColorNN(input_dim)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                # transform input for NN model
                if input_dim > 12:
                    x_expanded = np.zeros((1, input_dim), dtype=np.float32)
                    x_expanded[:, :12] = x_normalized  # copy original features
                    x_normalized = x_expanded
                
                # convert to tensor and predict
                x_tensor = torch.tensor(x_normalized, dtype=torch.float32)
                with torch.no_grad():
                    prediction = model(x_tensor).numpy()
                
                # scale back to 0-255 range
                prediction = prediction * 255.0
                print(f"Neural network prediction: {prediction}")
                return prediction[0]
            except Exception as e:
                print(f"Error using neural network model: {str(e)}")
                import traceback
                traceback.print_exc()
                # return a default non-zero value - for debugging
                return [100, 100, 100]
        else:
            print(f"Model file not found at {model_path}")
            # return a default non-zero value - for debugging
            return [100, 100, 100]
    else:
        # traditional ML models
        try:
            # dummy data for model loading
            dummy_train = np.zeros((10, 15), dtype=np.float32)
            dummy_val = np.zeros((5, 15), dtype=np.float32)
            dummy_test = np.zeros((1, 15), dtype=np.float32)
            dummy_test_df = pd.DataFrame()
            
            model = get_model(model_name, dummy_train, dummy_val, dummy_test, dummy_test_df)
            
            # make sure we're using the right number of features
            input_features = x_normalized if x_normalized.shape[1] <= 12 else x_normalized[:, :12]
            
            prediction = model.predict(input_features)
            
            # scale back to 0-255 range
            prediction = prediction * 255.0
            print(f"{model_name} prediction: {prediction}")
            return prediction[0]
        except Exception as e:
            print(f"Error using {model_name} model: {str(e)}")
            import traceback
            traceback.print_exc()
            # return a default non-zero value - for debugging
            return [100, 100, 100]
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # extract RGB values from the image
            rgb_list, colors, result_filename = extract_rgb_codes(filepath)
            
            objects_path = os.path.join('static', 'results', f'objects_{result_filename}')
            colors_path = os.path.join('static', 'results', f'colors_{result_filename}')
            
            visualize_detected_objects(objects_path)
            visualize_extracted_colors(colors, colors_path)
            
            return jsonify({
                'success': True,
                'rgb_list': rgb_list.tolist() if isinstance(rgb_list, np.ndarray) else rgb_list,
                'result_image': f'results/result_{result_filename}',
                'objects_image': f'results/objects_{result_filename}',
                'colors_image': f'results/colors_{result_filename}'
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Processing error: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    rgb_list = data.get('rgb_list')
    model_name = data.get('model')
    true_rgb = data.get('true_rgb', [0, 0, 0])
    
    print(f"Received prediction request for model: {model_name}")
    print(f"RGB list: {rgb_list}")
    
    if not rgb_list:
        return jsonify({'error': 'No RGB data provided'})
    
    try:
        # get prediction from selected model
        predicted_rgb = predict_color(rgb_list, model_name)
        
        if np.all(np.array(predicted_rgb) < 1.0):
            print("Warning: Prediction is all black, using fallback values")
            predicted_rgb = [128, 128, 128]  # Default to gray for debugging
        
        # calculate metrics if true values provided
        metrics = None
        if all(v != 0 for v in true_rgb):
            try:
                true_array = np.array([true_rgb], dtype=np.float32)
                pred_array = np.array([predicted_rgb], dtype=np.float32)
                metrics = evaluate_metrics(true_array, pred_array)
                
                metrics = {k: float(v) for k, v in metrics.items()}
            except Exception as e:
                print(f"Error calculating metrics: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # make sure predicted RGB values are rounded and in range 0-255
        predicted_rgb = [min(255, max(0, round(float(v)))) for v in predicted_rgb]
        
        print(f"Final predicted RGB: {predicted_rgb}")
        
        return jsonify({
            'success': True,
            'predicted_rgb': predicted_rgb,
            'metrics': metrics
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction error: {str(e)}'})
    
if __name__ == '__main__':
    app.run(debug=True)