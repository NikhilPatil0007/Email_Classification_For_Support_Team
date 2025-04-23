import os
import logging
from flask import Blueprint, request, jsonify
import torch
from utils import mask_pii

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables
os.environ["HF_HOME"] = "/tmp/huggingface"

# Create blueprint
api_bp = Blueprint("api", __name__)

# Global variables for model and tokenizer
tokenizer = None
model = None
device = None
id2label = {0: "Incident", 1: "Request", 2: "Problem", 3: "Change"}

def load_model():
    """Load model and tokenizer with error handling"""
    global tokenizer, model, device
    
    try:
        from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
        
        # Repo of Hugging Face Model Hub where Model is Pushed
        REPO_ID = "Nikpatil/Email_classifier"
        MAX_LENGTH = 256
        
        logger.info(f"Loading model from {REPO_ID}")
        
        # Use try/except for model loading to handle network issues
        try:
            tokenizer = DebertaV2Tokenizer.from_pretrained(REPO_ID)
            model = DebertaV2ForSequenceClassification.from_pretrained(REPO_ID)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Model loaded successfully. Using device: {device}")
            model.to(device)
            model.eval()
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"Error importing dependencies: {str(e)}")
        return False

# Load model when module is imported
model_loaded = load_model()

@api_bp.route("/classify", methods=["POST"])
def classify_email():
    """Endpoint to classify email content"""
    # Check if model is loaded
    global model_loaded
    if not model_loaded:
        model_loaded = load_model()
        if not model_loaded:
            return jsonify({"error": "Model initialization failed. Please try again later."}), 500
    
    # Get request data
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON payload"}), 400
    except Exception as e:
        logger.error(f"Error parsing JSON: {str(e)}")
        return jsonify({"error": "Invalid JSON format"}), 400
    
    # Validate email_body
    email_body = data.get("email_body")
    if email_body is None:
        return jsonify({"error": "Email body field is required"}), 400
    
    if not isinstance(email_body, str):
        return jsonify({"error": "Email body must be a string"}), 400
    
    if len(email_body.strip()) == 0:
        return jsonify({"error": "Email body cannot be empty"}), 400
    
    # Mask PII
    try:
        masked_email, entities = mask_pii(email_body)
    except Exception as e:
        logger.error(f"Error in PII masking: {str(e)}")
        return jsonify({"error": "Error processing email content"}), 500
    
    # Tokenize input
    try:
        inputs = tokenizer(
            masked_email,
            add_special_tokens=True,
            max_length=256,  # Using direct value for clarity
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception as e:
        logger.error(f"Error in tokenization: {str(e)}")
        return jsonify({"error": "Error processing email text"}), 500
    
    # Run inference
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        predicted_class_id = torch.argmax(probs).item()
        
        # Ensure the predicted class id is valid
        if predicted_class_id not in id2label:
            logger.error(f"Invalid predicted class ID: {predicted_class_id}")
            return jsonify({"error": "Model returned invalid prediction"}), 500
            
        predicted_class = id2label[predicted_class_id]
        
        # Format response
        response = {
            "input_email_body": email_body,
            "list_of_masked_entities": entities,
            "masked_email": masked_email,
            "category_of_the_email": predicted_class
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in model inference: {str(e)}")
        return jsonify({"error": "Error classifying email"}), 500

# Health check endpoint
@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify API status"""
    global model_loaded
    if not model_loaded:
        try:
            model_loaded = load_model()
        except:
            pass
    
    status = "healthy" if model_loaded else "unhealthy"
    return jsonify({
        "status": status,
        "model_loaded": model_loaded
    }), 200 if model_loaded else 503