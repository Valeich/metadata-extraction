import pypdf
import base64
import io
import os
import logging
import json
from flask import Flask, request, jsonify
from flask_cors import CORS  # Added for CORS support
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def extract_metadata_from_base64(encoded_pdf: str):
    """Extracts metadata from a base64-encoded PDF."""
    try:
        pdf_bytes = base64.b64decode(encoded_pdf)
        pdf_stream = io.BytesIO(pdf_bytes)
        reader = pypdf.PdfReader(pdf_stream)
        metadata = reader.metadata
        
        return {"metadata": {k: str(v) for k, v in metadata.items()} if metadata else "No metadata found"}
    except Exception as e:
        logger.error(f"Error processing PDF metadata: {e}")
        return {"error": f"Error reading PDF metadata: {e}"}

def generate_extraction_from_base64(encoded_pdf: str):
    """Uses Google Gen AI to extract structured data from a base64-encoded PDF."""
    try:
        pdf_bytes = base64.b64decode(encoded_pdf)
    except Exception as e:
        logger.error(f"Base64 decoding error: {e}")
        return {"error": f"Invalid base64 encoding: {e}"}

    try:
        # Ensure proper credentials are set via the environment (GOOGLE_APPLICATION_CREDENTIALS)
        client = genai.GenerativeModel("gemini-2.0-flash-exp")
        part = types.Part.from_bytes(bytes=pdf_bytes, mime_type="application/pdf")
        text_part = types.Part.from_text(text="Generate JSON output with only the parsed content.")

        system_instruction = (
            "I will give you PDF and Image files. The files are an official document that has the document number, "
            "the guidelines, and the details of a person to be hired as a civil servant. I need you to parse the civil "
            "person's information into JSON in the array format:\n"
            '{"name": "", "nip": "", "place_of_birth": "", "date_of_birth": "", "education": "", "title": "", '
            '"work_duration": "", "work_unit": "", "gov_instance": "", "signer": "", "signer_employee_id": "", "copied": ""}\n\n'
            "Detect if the document is photocopied by scanning all of the pages in the document, grayscale is a sign "
            "that the document is copied and return with yes or no.\n\n"
            "Do NOT add any other attributes. Do not hallucinate, if you cannot parse the text from the document "
            "respond with null. If the extraction process is successful return with status code 200, otherwise return with status code 400."
        )

        response = client.generate_content(
            contents=[types.Content(role="user", parts=[part, text_part])],
            config=types.GenerateContentConfig(
                temperature=0.1,
                top_p=0.95,
                max_output_tokens=8192,
                response_modalities=["TEXT"],
                system_instruction=system_instruction,
            ),
        )

        extraction_result = json.loads(response.text) if response.text else {"error": "Could not parse AI response"}
        return extraction_result
    except Exception as e:
        logger.error(f"Error processing AI extraction: {e}")
        return {"error": f"Error processing AI extraction: {e}"}

@app.route('/metadata', methods=['POST'])
def metadata_endpoint():
    """API endpoint to extract metadata from a PDF."""
    try:
        data = request.get_json()
        if not data or "base64_pdf" not in data:
            return jsonify({"error": "Missing 'base64_pdf' in request"}), 400

        metadata_result = extract_metadata_from_base64(data["base64_pdf"])
        return jsonify(metadata_result), 200
    except Exception as e:
        logger.error(f"Error processing /metadata: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/extract', methods=['POST'])
def extract_endpoint():
    """API endpoint to extract data from a PDF using Google Gen AI."""
    try:
        data = request.get_json()
        if not data or "base64_pdf" not in data:
            return jsonify({"error": "Missing 'base64_pdf' in request"}), 400

        extraction_result = generate_extraction_from_base64(data["base64_pdf"])
        return jsonify(extraction_result), 200
    except Exception as e:
        logger.error(f"Error processing /extract: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Cloud Run readiness."""
    return jsonify({"status": "running"}), 200

if __name__ == '__main__':
    # Cloud Run provides the PORT environment variable
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting application on port {port}")
    app.run(host='0.0.0.0', port=port)
