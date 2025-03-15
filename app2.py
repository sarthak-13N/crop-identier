import os
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify

# Initialize Flask App
app = Flask(__name__)

# Google Drive Model Link (Direct Download)
MODEL_URL = "https://drive.google.com/uc?id=16T2Nz28Y6k-tk27Pac75IIuzuiBCVjpU&export=download"
MODEL_PATH = "crop_identy.h5"

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading Model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model Downloaded Successfully!")

# Load Trained Model
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model Loaded Successfully!")

# Function to classify images
def classify_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        return "✅ Crop Plant" if prediction[0] <= 0.5 else "❌ Not a Crop"

    except Exception as e:
        return f"Error: {e}"

# API Route
@app.route("/identify", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    file_path = "temp_image.jpg"
    file.save(file_path)

    result = classify_image(file_path)
    os.remove(file_path)

    return jsonify({"prediction": result})

# Run Flask Server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
