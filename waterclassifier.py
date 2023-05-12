from flask import Flask,jsonify,request
import numpy as np
from PIL import Image
from flask_cors import CORS
import tensorflow as tf
from io import BytesIO
import base64

model = tf.keras.models.load_model('model.h5')

labels = {
    0: 'No Microbes Found',
    1: 'Microbes Found'
}

app = Flask(__name__)
CORS(app)

@app.route("/type/",methods=['POST'])
def return_type():
    img = request.get_data()
    img = np.asarray(Image.open(BytesIO(base64.b64decode(img))).resize((100,100)).convert("RGB"), dtype=np.float32)
    img = np.expand_dims(img, axis = 0)
    prediction = model.predict(img)
    confidence = np.max(prediction) * 100
    confidence = f"{confidence:.2f}"
    prediction = np.argmax(prediction)
    prediction = labels[prediction]
    result = {'type': prediction, 'confidence': confidence}
    return result

@app.route("/",methods=['GET'])
def default():
  return "<h1>You shouldn't be here</h1>"

if __name__ == "__main__":
    app.run() 