# Email Classification & PII Processing System for IT Support Automation

## Introduction
This API classifies support emails into four ITIL categories (Incident, Request, Problem, Change) using a fine-tuned DeBERTa model. The service automatically masks personally identifiable information (PII) before processing, ensuring privacy compliance.

## Features
- **Automatic Classification**: Fast categorization of support emails
- **PII Protection**: Automatic masking of sensitive information
- **ITIL Alignment**: Categories map to standard ITIL service management framework
- **High Accuracy**: Based on state-of-the-art DeBERTa language model
- **Easy Integration**: Simple REST API interface

## Hardware Requirements
* **For Development/Testing:**
   * CPU: 2+ cores
   * RAM: 4GB minimum (8GB recommended)
   * Storage: 1GB free space
* **For Production Deployment:**
   * CPU: 4+ cores
   * RAM: 8GB minimum (16GB recommended for higher throughput)
   * GPU: Optional but beneficial for faster inference

## Software Requirements
* Python 3.8+
* Flask
* PyTorch
* Transformers library
* PII masking utilities


## How to Run

### Local Development

```bash
# Clone repository
git clone <repository-url>
cd email-classification-api

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Hugging Face Spaces Deployment
1. The API is already deployed at: https://huggingface.co/spaces/Nikpatil/Email_Classification_For_Support_Team
2. No additional setup required to use the deployed version

## How to Access

### Using the API Endpoint
Send POST requests to: `https://nikpatil-email-classification-for-support-team.hf.space/classify`


## 📄 Technical Report

You can find the detailed technical report [here](/Technical_Report.pdf).


## Privacy & Security
- All PII is masked before processing
- No data is stored after processing
- Compliant with standard data protection practices

