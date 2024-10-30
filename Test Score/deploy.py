import pickle
from flask import Flask, request, json, jsonify
import numpy as np
app = Flask(__name__)
filename = 'knn_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
@app.route('/diabetes/v1/predict', methods=['POST'])
def predict():
 features = request.json
 features_list = [features["ChuyenCan"], features["BTL"], features['GiuaKi']]

 prediction = loaded_model.predict([features_list])
 response = {}
 response['prediction'] = float(prediction[0])
 return jsonify(response)

if __name__ == '__main__':
 app.run(host='0.0.0.0', port=5000)