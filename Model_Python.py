import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# === CHANGE YOUR PASSWORD HERE ===
APP_PASSWORD = "Projura2024!"
# ================================

class NeuralNetwork:
    def __init__(self):
        self.inputs_number = 4
        self.inputs_names = [
            'No of Beds_lag_0', 
            'No of Units_lag_0', 
            'Internal Floor Area_lag_0', 
            'Contractor Related Risks_lag_0'
        ]

    def calculate_outputs(self, inputs):
        No_of_Beds_lag__zero_ = inputs[0]
        No_of_Units_lag__zero_ = inputs[1]
        Internal_Floor_Area_lag__zero_ = inputs[2]
        Contractor_Related_Risks_lag__zero_ = inputs[3]

        scaled_No_of_Beds_lag__zero_ = (No_of_Beds_lag__zero_ - 189.1679993) / 257.9400024
        scaled_No_of_Units_lag__zero_ = (No_of_Units_lag__zero_ - 105.2959976) / 157.5339966
        scaled_Internal_Floor_Area_lag__zero_ = (Internal_Floor_Area_lag__zero_ - 7812.939941) / 11333.90039
        scaled_Contractor_Related_Risks_lag__zero_ = (Contractor_Related_Risks_lag__zero_ - 0.5211380124) / 0.0262020994

        perceptron_layer_1_output_0 = np.tanh(
            -0.262251 + (scaled_No_of_Beds_lag__zero_ * 1.33131) +
            (scaled_No_of_Units_lag__zero_ * 0.836534) +
            (scaled_Internal_Floor_Area_lag__zero_ * 0.899761) +
            (scaled_Contractor_Related_Risks_lag__zero_ * 0.0715323)
        )
        perceptron_layer_1_output_1 = np.tanh(
            -0.0471727 + (scaled_No_of_Beds_lag__zero_ * 0.0826954) +
            (scaled_No_of_Units_lag__zero_ * 0.0979729) +
            (scaled_Internal_Floor_Area_lag__zero_ * -0.129047) +
            (scaled_Contractor_Related_Risks_lag__zero_ * 0.0460621)
        )
        perceptron_layer_1_output_2 = np.tanh(
            1.0947 + (scaled_No_of_Beds_lag__zero_ * -0.245285) +
            (scaled_No_of_Units_lag__zero_ * 0.375173) +
            (scaled_Internal_Floor_Area_lag__zero_ * 0.182613) +
            (scaled_Contractor_Related_Risks_lag__zero_ * -0.0968)
        )
        perceptron_layer_1_output_3 = np.tanh(
            1.88753 + (scaled_No_of_Beds_lag__zero_ * 0.0557834) +
            (scaled_No_of_Units_lag__zero_ * 0.167401) +
            (scaled_Internal_Floor_Area_lag__zero_ * -0.478933) +
            (scaled_Contractor_Related_Risks_lag__zero_ * 0.133224)
        )
        perceptron_layer_1_output_4 = np.tanh(
            -1.23089 + (scaled_No_of_Beds_lag__zero_ * 0.172787) +
            (scaled_No_of_Units_lag__zero_ * -1.45926) +
            (scaled_Internal_Floor_Area_lag__zero_ * -0.512254) +
            (scaled_Contractor_Related_Risks_lag__zero_ * -0.0201704)
        )

        perceptron_layer_2_output_0 = (
            1.11418 +
            (perceptron_layer_1_output_0 * -0.0127957) +
            (perceptron_layer_1_output_1 * 3.47178) +
            (perceptron_layer_1_output_2 * 3.0903) +
            (perceptron_layer_1_output_3 * -4.35496) +
            (perceptron_layer_1_output_4 * -1.29104)
        )

        unscaling_layer_output_0 = perceptron_layer_2_output_0 * 21.09350014 + 31.92009926

        Project_Duration__Months__lag__zero_ = unscaling_layer_output_0
        out = [None] * 1

        out[0] = Project_Duration__Months__lag__zero_

        return out

    def calculate_batch_output(self, input_batch):
        output_batch = [None] * input_batch.shape[0]

        for i in range(input_batch.shape[0]):
            inputs = list(input_batch[i])
            output = self.calculate_outputs(inputs)
            output_batch[i] = output

        return output_batch

app = Flask(__name__)
CORS(app)
model = NeuralNetwork()

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Projura AI Flask backend is running!'})

# -------- Password check endpoint --------
@app.route('/check_password', methods=['POST'])
def check_password():
    data = request.get_json()
    if not data or 'password' not in data:
        return jsonify({"success": False, "message": "Missing password"}), 400

    if data['password'] == APP_PASSWORD:
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": "Incorrect password"}), 401

# -------- Estimation endpoint ------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    required = ['beds', 'units', 'area', 'risk']
    for key in required:
        if key not in data:
            return jsonify({'error': f'Missing key: {key}'}), 400
    try:
        inputs = [
            float(data['beds']),
            float(data['units']),
            float(data['area']),
            float(data['risk'])
        ]
    except Exception:
        return jsonify({'error': 'Invalid input values'}), 400
    output = model.calculate_outputs(inputs)
    return jsonify({'duration_months': output[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)