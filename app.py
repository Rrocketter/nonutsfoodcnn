from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from flask_cors import CORS  # Import Flask-CORS

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS support
model = tf.keras.models.load_model('models/foodnutidentifier.h5')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    
    # Preprocess the image
    image = Image.open(image)
    image = image.resize((256, 256))
    image = np.array(image) / 255.0  # Normalize the image
    
    # Perform inference
    predictions = model.predict(np.expand_dims(image, 0))
    result = predictions[0][0]  # Assuming binary classification
    
    prediction = float(result)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

