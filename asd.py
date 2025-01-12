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

# # Load model regresi
# REGRESSION_MODEL_PATH = 'vgg19_regression_model.h5'
# regression_model = load_model(REGRESSION_MODEL_PATH, custom_objects={'mse': MeanSquaredError()})

# Path model berdasarkan parameter
MODEL_PATHS = {
    'vgg': {
        '20x20': {'gee': 'model/vgg_20x20_gee.h5', 'drone': 'model/vgg_20x20_drone.h5', 'mix': 'model/vgg_20x20_mix.h5'},
        '10x10': {'gee': 'model/vgg_10x10_gee.h5', 'drone': 'model/vgg_10x10_drone.h5', 'mix': 'model/vgg_10x10_mix.h5'},
        '5x5': {'gee': 'model/vgg_5x5_gee.h5', 'drone': 'model/vgg_5x5_drone.h5', 'mix': 'model/vgg_5x5_mix.h5'},
        '1x1': {'gee': 'model/vgg_1x1_gee.h5', 'drone': 'model/vgg_1x1_drone.h5', 'mix': 'model/vgg_1x1_mix.h5'},
        'gabung': {'gee': 'model/vgg_gabung_gee.h5', 'drone': 'model/vgg_gabung_drone.h5', 'mix': 'model/vgg_gabung_mix.h5'}
    },
    'cnn': {
        '20x20': {'gee': 'model/cnn_20x20_gee.h5', 'drone': 'model/cnn_20x20_drone.h5', 'mix': 'model/cnn_20x20_mix.h5'},
        '10x10': {'gee': 'model/cnn_10x10_gee.h5', 'drone': 'model/cnn_10x10_drone.h5', 'mix': 'model/cnn_10x10_mix.h5'},
        '5x5': {'gee': 'model/cnn_5x5_gee.h5', 'drone': 'model/cnn_5x5_drone.h5', 'mix': 'model/cnn_5x5_mix.h5'},
        '1x1': {'gee': 'model/cnn_1x1_gee.h5', 'drone': 'model/cnn_1x1_drone.h5', 'mix': 'model/cnn_1x1_mix.h5'},
        'gabung': {'gee': 'model/cnn_gabung_gee.h5', 'drone': 'model/cnn_gabung_drone.h5', 'mix': 'model/cnn_gabung_mix.h5'}
    },
    'resnet': {
        '20x20': {'gee': 'model/resnet_20x20_gee.h5', 'drone': 'model/resnet_20x20_drone.h5', 'mix': 'model/resnet_20x20_mix.h5'},
        '10x10': {'gee': 'model/resnet_10x10_gee.h5', 'drone': 'model/resnet_10x10_drone.h5', 'mix': 'model/resnet_10x10_mix.h5'},
        '5x5': {'gee': 'model/resnet_5x5_gee.h5', 'drone': 'model/resnet_5x5_drone.h5', 'mix': 'model/resnet_5x5_mix.h5'},
        '1x1': {'gee': 'model/resnet_1x1_gee.h5', 'drone': 'model/resnet_1x1_drone.h5', 'mix': 'model/resnet_1x1_mix.h5'},
        'gabung': {'gee': 'model/resnet_gabung_gee.h5', 'drone': 'model/resnet_gabung_drone.h5', 'mix': 'model/resnet_gabung_mix.h5'}
    }
}

# Fungsi untuk menangani prediksi berdasarkan model type
def handle_prediction(model_type):
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    size = request.args.get('size')
    data_type = request.args.get('type')

    if file.filename == '' or not size or not data_type:
        return jsonify({'error': 'Missing file or parameters (size, type)'}), 400

    try:
        # Simpan file sementara
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)

        # Proses gambar dan buat prediksi
        image = prepare_image(file_path)
        prediction = predict_regression(model_type, size, data_type, image)

        # Hapus file sementara
        os.remove(file_path)

        return jsonify({'predicted_value': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


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

# Fungsi umum untuk prediksi regresi
def predict_regression(model_type, size, data_type, image):
    try:
        model_path = MODEL_PATHS[model_type][size][data_type]
        model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
        prediction = model.predict(image)[0][0]
        final_prediction = (prediction / 40) + 8
        return f"{final_prediction:.2f} ton/ha"
    except KeyError:
        raise ValueError("Invalid parameters. Please check the model_type, size, or data_type.")

# Endpoint untuk prediksi regresi (VGG)
@app.route('/predict_vgg', methods=['POST'])
def predict_vgg():
    return handle_prediction('vgg')

# Endpoint untuk prediksi regresi (CNN)
@app.route('/predict_cnn', methods=['POST'])
def predict_cnn():
    return handle_prediction('cnn')

# Endpoint untuk prediksi regresi (ResNet)
@app.route('/predict_resnet', methods=['POST'])
def predict_resnet():
    return handle_prediction('resnet')

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