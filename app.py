import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
from PIL import Image
import torchvision.transforms as transforms
from src.config import Config
from src.model import get_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = Config.DEVICE
model =  get_model(
    num_classes=Config.NUM_CLASSES, 
    pretrained=False,
    model_name=Config.MODEL_NAME  # ✅ Add this
)
checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print("Model loaded successfully!")

transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=Config.MEAN, std=Config.STD)
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        print(f"LOGITS: {outputs}")
        probs = torch.softmax(outputs, dim=1)
        print(f"PROBS: {probs}")
        confidence, predicted = torch.max(probs, 1)
        predicted_class = predicted.item()
        class_name = Config.CLASS_NAMES[predicted_class]
    
    return {
        'class': class_name,
        'confidence': f"{confidence.item()*100:.1f}%",
        'probs': {Config.CLASS_NAMES[i]: f"{probs[0][i].item()*100:.1f}%" for i in range(2)}
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    result = predict_image(filepath)
    result['image_path'] = f"/static/uploads/{filename}"
    return jsonify(result)

if __name__ == '__main__':
    print(f"Using device: {Config.DEVICE}")
    app.run(debug=True, host='0.0.0.0', port=5000)
