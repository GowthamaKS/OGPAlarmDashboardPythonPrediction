from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Incident Forecast API is running!"})

@app.route("/train", methods=["POST"])
def train():
    try:
        # train_model()  # Uncomment when actual model is available
        return jsonify({"message": "Model trained successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data or 'timestamp' not in data or 'days' not in data:
            return jsonify({"error": "Missing timestamp or days"}), 400

        current_timestamp = data.get("timestamp")
        num_days = int(data.get("days", 7))

        # ✅ Validate timestamp format: "YYYY-MM-DD HH:MM:SS"
        try:
            start_date = datetime.strptime(current_timestamp, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return jsonify({"error": "Invalid timestamp format. Use YYYY-MM-DD HH:MM:SS"}), 400

        predictions = predict_future_incidents(current_timestamp, num_days=num_days)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/device-alerts", methods=["GET"])
def device_alerts():
    try:
        # alerts = get_device_alerts()  # Uncomment when actual function is available
        alerts = []  # Dummy response
        return jsonify(alerts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Dummy predict_future_incidents function
def predict_future_incidents(timestamp, num_days):
    start_date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    predictions = []
    for i in range(num_days):
        date = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
        predictions.append({
            'date': date,
            'predicted_incident_count': i + 1  # Replace with actual ML model prediction
        })
    return predictions

if __name__ == "__main__":
    # ✅ Fix signal issue on Windows
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
