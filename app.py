from flask import Flask, jsonify, request
from flask_cors import CORS
from model import train_model, predict_future_incidents, get_device_alerts, get_incident_summary
from datetime import datetime
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Corrected logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Incident Forecast API is running!"})

@app.route("/train", methods=["POST"])
def train():
    try:
        train_model()
        logger.info("Model trained successfully")
        return jsonify({"message": "Model trained successfully!"})
    except Exception as e:
        logger.error(f"Error training model: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or 'timestamp' not in data or 'days' not in data:
            logger.error("Invalid request payload: missing timestamp or days")
            return jsonify({"error": "Request must include 'timestamp' and 'days'"}), 400

        current_timestamp = data['timestamp']
        num_days = int(data['days'])

        try:
            datetime.strptime(current_timestamp, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            logger.error(f"Invalid timestamp format: {current_timestamp}")
            return jsonify({"error": "Timestamp must be in format 'YYYY-MM-DD HH:MM:SS'"}), 400

        predictions = predict_future_incidents(current_timestamp, num_days=num_days)
        logger.info(f"Returning predictions: {predictions}")
        return jsonify(predictions)
    except ValueError as ve:
        logger.error(f"ValueError in predict: {str(ve)}", exc_info=True)
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in predict: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/device-alerts", methods=["GET"])
def device_alerts():
    try:
        alerts = get_device_alerts()
        logger.info(f"Returning device alerts: {alerts}")
        return jsonify(alerts)
    except Exception as e:
        logger.error(f"Error in device-alerts: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/incident-summary", methods=["GET"])
def incident_summary():
    try:
        summary = get_incident_summary()
        logger.info(f"Returning incident summary: {summary}")
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error in incident-summary: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)