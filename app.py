import logging
from flask import Flask, jsonify
from api import api_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create and configure Flask app
app = Flask(__name__)
app.register_blueprint(api_bp)

# Adding error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def server_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "message": "Email Classification API is running. Use the /classify endpoint to classify emails."
    })

if __name__ == "__main__":
    logger.info("Starting Email Classification API server...")
    app.run(host="0.0.0.0", port=7860)