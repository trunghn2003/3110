import pickle
from flask import Flask, request, json, jsonify
import numpy as np
app = Flask(__name__)

filename = 'linear_svm_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

@app.route('/diabetes/v1/predict', methods=['POST'])
def predict():
 features = request.json
 features_list = [features["cp"],
 features["thalachh"], features["slp"]]

 prediction = loaded_model.predict([features_list])
 confidence = loaded_model.predict_proba([features_list])
 response = {}
 response['prediction'] = int(prediction[0])
 response['confidence'] = str(round(np.amax(confidence[0]) * 100 ,2))
 return jsonify(response)

if __name__ == '__main__':
 app.run(host='0.0.0.0', port=5000)