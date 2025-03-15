import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
import os
import gdown  # Import gdown to download files from Google Drive

# Google Drive File ID (Replace with your actual file ID)
FILE_ID = "16T2Nz28Y6k-tk27Pac75IIuzuiBCVjpU"
MODEL_PATH = "crop_identy.h5"

# Function to Download Model if Not Present
def download_model():
    if not os.path.exists(MODEL_PATH):  # Check if the model already exists
        print("📥 Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Model Downloaded Successfully!")
    else:
        print("✅ Model already exists, skipping download.")



 # Download the model before loading
download_model()

# Load the Trained Model
print("🔄 Loading Model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model Loaded Successfully!")
   
# Initialize Flask App
app = Flask(__name__)

# Load Trained Model
#model = tf.keras.models.load_model("crop_identy.h5")
#print("✅ Model Loaded Successfully!")

# Function to Classify Image
def classify_image(img_path):
    try:
        # Load and Preprocess Image
        img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input
        img_array = image.img_to_array(img)  # Convert to array
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize and add batch dimension

        # Predict
        prediction = model.predict(img_array)
        print(f"🔍 Raw Prediction Output: {prediction[0]}")  # Debugging Output

        # Adjust Condition Based on Model Training
        if prediction[0] > 0.5:  # Assuming 1 = Non-Crop, 0 = Crop
            return "❌ Not a Crop"
        else:
            return "✅ Crop Plant"

    except Exception as e:
        return f"Error: {e}"  # Handles issues like incorrect file paths

# API Route for Image Classification
@app.route("/identify", methods=["POST"])
def predict():
    try:
        # Check if file is uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]
        print(f"✅ Received File: {file.filename}")  # Debugging Output
        
        # Save Image Temporarily
        file_path = "temp_image.jpg"
        file.save(file_path)

        # Predict Image Category
        result = classify_image(file_path)

        # Delete Temporary Image
        os.remove(file_path)

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask Server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port or default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)
