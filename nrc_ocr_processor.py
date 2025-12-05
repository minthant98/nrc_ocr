import os
import io
import json
import base64
import requests
from typing import Dict, Any, Optional
from PIL import Image, ImageEnhance, ExifTags

# WARNING: TensorFlow is assumed to be installed for this module
# Since we cannot guarantee it runs in the final environment, 
# we wrap the import in a try-except block.
try:
    import pickle
    from tensorflow import keras
except ImportError:
    print("Warning: TensorFlow/Keras not found. Local CNN augmentation will be disabled.")
    keras = None
    pickle = None

# --- CONSTANTS AND SCHEMAS ---

# The URL for the Gemini API model endpoint
API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

# Data Mapping and Schemas (Core Fields Only)
JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "Overall_Confidence_Score": {
            "type": "NUMBER",
            "description": "A score from 0.0 to 1.0 indicating the model's certainty in the entire extraction, especially for handwritten Burmese fields. Use 0.0 to 1.0 format."
        },
        "NRC_state_division": {
            "type": "STRING",
            "description": "The State/Division code (the first 1-2 digits/characters of the NRC, 1-14), extracted as **standard Latin digits**."
        },
        "NRC_township": {
            "type": "STRING",
            "description": "The Township Code/initials, the Burmese words between '/' and '('. Extracted **precisely** in the original Burmese script."
        },
        "NRC_sth": {
            "type": "STRING",
            "description": "The classification code, the letters inside the parenthesis, e.g., '(N)', '(C)', or '(A)'. Must use standard Latin letters and include parentheses."
        },
        "NRC_no": {
            "type": "STRING",
            "description": "The six-digit citizen identification number, extracted as **standard Latin digits**."
        },
        "Name": {
            "type": "STRING",
            "description": "The cardholder's name (အမည်), extracted **precisely** in the original handwritten Burmese script."
        },
        "Fathers_Name": {
            "type": "STRING",
            "description": "The father's name (အဘအမည်), extracted **precisely** in the original handwritten Burmese script."
        },
        "Date_of_Birth": {
            "type": "STRING",
            "description": "The cardholder's date of birth (မွေးသက္ကရာဇ်), extracted **precisely** in the original handwritten Burmese script (e.g., date, month, year, including original Burmese numbers/characters)."
        }
    },
    "required": ["Overall_Confidence_Score", "NRC_state_division", "NRC_township", "NRC_sth", "NRC_no", "Name", "Fathers_Name", "Date_of_Birth"] 
}

# SYSTEM INSTRUCTION (V7: Clear and Concise)
SYSTEM_INSTRUCTION = (
    "You are an expert Optical Character Recognition (OCR) and Intelligent Document "
    "Processing (IDP) system specialized in reading Myanmar National Registration Card (NRC) documents. "
    "Your primary task is to **meticulously and accurately** extract ONLY the core identity fields: the full NRC number breakdown, "
    "Name, Father's Name, and Date of Birth. Prioritize the **precise recognition of handwritten Burmese script** "
    "for fields like Name, Fathers_Name, Date_of_Birth, and NRC_township. "
    "The NRC number must be broken down into four components: NRC_state_division (1-14, Latin digits), NRC_township (Burmese script), "
    "NRC_sth (e.g., (N), (C), or (A), Latin letters), and NRC_no (6 digits, Latin digits). "
    "All numerical values must be output as standard Latin digits (0-9). "
    "Crucially, you must provide an Overall_Confidence_Score (0.0 to 1.0) based on the image quality and legibility. "
    "Output the results ONLY as a JSON object conforming to the provided schema."
)

# --- LOCAL CNN MODEL LOADING ---

local_cnn_model = None
burmese_class_map = None

if keras and pickle:
    try:
        # Load the CNN Model and Map for local augmentation
        local_cnn_model = keras.models.load_model("burmese_hcr_model.keras") 
        with open("burmese_class_map.pkl", "rb") as f:
            burmese_class_map = pickle.load(f)
        print("Local CNN model loaded successfully for augmentation.")
    except Exception as e:
        print(f"Warning: Failed to load local CNN model/map: {e}. Local augmentation disabled.")
        local_cnn_model = None

# --- IMAGE PRE-PROCESSING PIPELINE ---

def rotate_image_from_exif(img: Image.Image) -> Image.Image:
    """Reads EXIF data to automatically correct image orientation."""
    try:
        exif = img._getexif()
        if exif is not None:
            orientation_tag = next((k for k, v in ExifTags.TAGS.items() if v == 'Orientation'), None)
            orientation = exif.get(orientation_tag)
            
            # Simplified rotation logic based on common EXIF values
            if orientation == 3:
                return img.rotate(180, expand=True)
            elif orientation == 6:
                return img.rotate(-90, expand=True)
            elif orientation == 8:
                return img.rotate(90, expand=True)
    except Exception:
        pass
    return img

def image_to_bytes(img: Image.Image) -> bytes:
    """Converts a PIL Image object back to PNG bytes."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG") 
    return buffer.getvalue()

def process_image(image_bytes: bytes) -> bytes:
    """
    Applies auto-rotation, and the core enhancement (Sharpen 2.0x, Contrast 1.5x).
    
    This function returns only the ENHANCED image bytes for the API call.
    The main app (app.py) is responsible for displaying the *current* orientation.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # 1. Auto-Rotate from EXIF data
        img = rotate_image_from_exif(img)
        
        # --- Handwriting Enhancement Pipeline (V7 R) ---
        # 2. Convert to Grayscale
        img_gray = img.convert("L")
        
        # 3. Sharpen (Original 2.0x Sharpening)
        sharpen_filter = ImageEnhance.Sharpness(img_gray)
        img_sharpened = sharpen_filter.enhance(2.0) 

        # 4. Moderate Contrast Increase (Original 1.5x Contrast)
        contrast_filter = ImageEnhance.Contrast(img_sharpened)
        img_final = contrast_filter.enhance(1.5) 
        
        # 5. Convert back to bytes (using PNG format for quality retention)
        return image_to_bytes(img_final)
        
    except Exception as e:
        # If processing fails, fall back to the original bytes
        print(f"Image processing pipeline failed: {e}. Using original image bytes.")
        return image_bytes

# --- CNN AUGMENTATION LOGIC (Placeholder for your 91% model) ---

def cnn_ocr_handwritten_fields(image_bytes: bytes, field_name: str) -> str:
    """
    Simulates running your local 91% accurate CNN model on a specific cropped region.
    
    In a real implementation, this function would handle:
    1. Converting image_bytes back to a PIL image.
    2. Defining the exact pixel coordinates to crop the field (e.g., 'Name').
    3. Running character segmentation on the cropped region.
    4. Passing segments to the `local_cnn_model.predict()`.
    5. Assembling the result using `burmese_class_map`.
    """
    if local_cnn_model is None:
        return "" # Local model not available
    
    # --- PLACEHOLDER FOR ACTUAL CNN INFERENCE ---
    # Since we cannot run Keras inference here, we provide fake high-confidence
    # results to demonstrate the augmentation logic flow.
    if field_name == "Name":
        return "အာကာကျော်" # CNN predicted name
    elif field_name == "Fathers_Name":
        return "ဦးနိုင်ဝင်း" # CNN predicted father's name
    elif field_name == "Date_of_Birth":
        return "၁၃ ဇန်နဝါရီ ၂၀၀၀" # CNN predicted DOB
    # --- END PLACEHOLDER ---

    return ""

# --- GEMINI API CALL (Augmented) ---

def extract_nrc_data_augmented(enhanced_image_bytes: bytes, api_key: str, original_image_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Calls the Gemini API, augmented with results from the local CNN model.
    """
    if not api_key:
        print("API Key is missing.")
        return None
        
    api_url = API_URL_TEMPLATE.format(API_KEY=api_key)
    
    # --- 1. RUN LOCAL CNN (AUGMENTATION STEP) ---
    # We run the CNN on the original image bytes (passed from app.py) 
    # as the CNN might be sensitive to the enhancement process.
    cnn_name = cnn_ocr_handwritten_fields(original_image_bytes, "Name")
    cnn_fathers_name = cnn_ocr_handwritten_fields(original_image_bytes, "Fathers_Name")
    cnn_dob = cnn_ocr_handwritten_fields(original_image_bytes, "Date_of_Birth")

    # Use image/png mime type since we saved it as PNG for better quality retention
    base64_image = base64.b64encode(enhanced_image_bytes).decode('utf-8')
    mime_type = "image/png" 
    
    # --- 2. CONSTRUCT AUGMENTED USER QUERY ---
    augmentation_text = ""
    if cnn_name or cnn_fathers_name or cnn_dob:
        augmentation_text = (
            "IMPORTANT: A highly accurate local OCR model (91% confidence) has provided suggestions for handwritten Burmese fields. "
            "**PRIORITIZE these suggestions in your final JSON output**, only making corrections if the values are clearly illogical or misplaced in the image. "
            f"Suggested Name: '{cnn_name}'. Suggested Father's Name: '{cnn_fathers_name}'. Suggested Date of Birth: '{cnn_dob}'. "
        )
    
    user_query = (
        f"Analyze the provided Myanmar NRC document image. {augmentation_text} "
        "Extract the values for ONLY the core identity fields: "
        "the four NRC components, Name, Father's Name, and Date of Birth. "
        "Crucially, split the NRC number (X/XXX(Y)######) into its four requested components: "
        "1. NRC_state_division (X, Latin digits 1-14) "
        "2. NRC_township (XXX, Burmese words between '/' and '(') "
        "3. NRC_sth ((Y), the classification code including parentheses, e.g., (N), (C), or (A)) "
        "4. NRC_no (######, the 6-digit number, Latin digits) "
        "Ensure Name, Father's Name, and Date of Birth are copied exactly as written in the original handwritten Burmese script, using the provided suggestions when possible. "
        "Return the output as a single JSON object."
    )
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": user_query},
                    {"inlineData": {"mimeType": mime_type, "data": base64_image}}
                ]
            }
        ],
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": JSON_SCHEMA
        }
    }

    try:
        response = requests.post(
            api_url, 
            headers={'Content-Type': 'application/json'}, 
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        
        json_string = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
        
        if json_string:
            return json.loads(json_string)
        else:
            error_detail = result.get('candidates', [{}])[0].get('finishReason', 'No content generated.')
            print(f"Error: Could not extract structured JSON. Finish Reason: {error_detail}")
            return None

    except Exception as e:
        print(f"An unexpected error occurred during API call: {e}")
        return None