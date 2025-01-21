from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

# Class labels corresponding to the model's output
class_labels = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']

# Load the trained model
import pickle

with open("SkinCancerPrediction.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Request method:", request.method)
        print("Request files:", request.files)
        # Check if an image file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        # Open the image and preprocess it
        image = Image.open(file).resize((180, 180))  # Adjust size if needed
        
        image_array = np.array(image)   # Normalize the image / 255.0
        image_array = image_array.reshape((1, 180, 180, 3))

        # Make a prediction
        prediction = model.predict(image_array)
        print(prediction);
         # Get the index of the highest probability
        predicted_class_index = np.argmax(prediction)

        # Get the label corresponding to the predicted class
        predicted_class = class_labels[predicted_class_index]

        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
