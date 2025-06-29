from flask import Flask, request, jsonify
import os
from PIL import Image
import torch
from torchvision import transforms
import librosa
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Загрузка моделей (положи файлы в одну папку)
camera_model = torch.load('camera_model.pt', map_location=torch.device('cpu'))
camera_model.eval()
sound_model = load_model('sound_model.h5')

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

state = {"sluna": False, "bark": False, "disoriented": False}

@app.route('/analyze_photo', methods=['POST'])
def analyze_photo():
    photo = request.files['photo']
    img = Image.open(photo.stream).convert('RGB')
    x = image_transform(img).unsqueeze(0)
    with torch.no_grad():
        output = camera_model(x)
    pred = torch.softmax(output, dim=1)[0]
    label = torch.argmax(pred).item()
    classes = ['normal', 'sluna', 'seizure']
    label_name = classes[label]
    if label_name == 'sluna': state['sluna'] = True
    if label_name == 'seizure': state['disoriented'] = True
    return jsonify({"camera_result": label_name, "probabilities": pred.tolist()})

@app.route('/analyze_sound', methods=['POST'])
def analyze_sound():
    audio = request.files['audio']
    y, sr = librosa.load(audio.stream, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    pred = sound_model.predict(mfcc)
    confidence = float(pred[0, 0])
    label = 'bark' if confidence > 0.5 else 'other'
    if label == 'bark': state['bark'] = True
    return jsonify({"sound_result": label, "confidence": confidence})

@app.route('/final_check', methods=['GET'])
def final_check():
    if state['sluna'] and state['bark'] and state['disoriented']:
        return jsonify({"status": "🚨 Возможно БЕШЕНСТВО!"})
    return jsonify({"status": "✅ Нет критических признаков"})

@app.route('/reset_state', methods=['POST'])
def reset_state():
    for k in state: state[k] = False
    return jsonify({"status": "state reset"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
