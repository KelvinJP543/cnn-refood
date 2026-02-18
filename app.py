import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tflite_runtime.interpreter as tflite

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)

# Load TFLite Model (Gunakan file hasil download dari Colab)
model_path = "model_quantized.tflite"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Inisialisasi Interpreter TFLite
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Mendapatkan detail input dan output model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/')
def home():
    return jsonify({
        "message": "API is running with TFLite. Use the /predict endpoint to classify images."
    })

@app.route('/predict', methods=['POST'])
def predict():
    # Periksa apakah ada file yang diupload
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    try:
        # Preprocessing gambar langsung menggunakan Pillow (Lebih ringan dari keras.preprocessing)
        img = Image.open(file).convert('RGB')
        img = img.resize((224, 224))
        
        # Konversi ke array dan normalisasi
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Melakukan Prediksi dengan TFLite
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # Klasifikasi berdasarkan threshold 0.5
        class_label = 'Hewani' if prediction[0][0] < 0.5 else 'Nabati'

        return jsonify({
            "class_label": class_label,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Jalankan aplikasi (Render akan mengatur port secara otomatis via environment variable)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)