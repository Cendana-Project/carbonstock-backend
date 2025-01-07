from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import os

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Load model klasifikasi
CLASSIFICATION_MODEL_PATH = 'carbon_stock_model_vgg19.h5'
classification_model = load_model(CLASSIFICATION_MODEL_PATH)

# Load model regresi
REGRESSION_MODEL_PATH = 'vgg19_regression_model.h5'
regression_model = load_model(REGRESSION_MODEL_PATH, custom_objects={'mse': MeanSquaredError()})

# Label kelas (ubah sesuai label dataset Anda)
class_labels = ['Rendah', 'Sedang', 'Tinggi']

# Fungsi untuk memproses gambar
def prepare_image(image_path, target_size=(224, 224)):
    try:
        image = load_img(image_path, target_size=target_size)  # Load gambar dengan ukuran target
        image = img_to_array(image)  # Konversi gambar ke array numpy
        image = np.expand_dims(image, axis=0)  # Tambahkan dimensi batch
        image = image / 255.0  # Normalisasi nilai piksel
        return image
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

# Endpoint untuk klasifikasi gambar
@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Simpan file sementara
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)

        # Proses gambar dan buat prediksi
        image = prepare_image(file_path)
        predictions = classification_model.predict(image)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Hapus file sementara
        os.remove(file_path)

        return jsonify({
            'class': predicted_class,
            'confidence': float(confidence)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint untuk regresi gambar
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Simpan file sementara
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)

        # Proses gambar dan buat prediksi
        image = prepare_image(file_path)
        prediction = regression_model.predict(image)[0][0]  # Hasil prediksi

        # Modifikasi hasil prediksi
        final_prediction = (prediction / 40) + 8
        formatted_prediction = f"{final_prediction:.2f} ton/ha"

        # Hapus file sementara
        os.remove(file_path)

        return jsonify({
            'predicted_value': formatted_prediction
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Jalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)