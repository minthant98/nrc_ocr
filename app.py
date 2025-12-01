import streamlit as st
import requests
import base64
import json
import io
import os 
from typing import Dict, Any, List, Optional
import difflib 
from datetime import date
from PIL import Image, ImageEnhance, ExifTags # Import PIL components

# --- Configuration for Gemini API ---
# The code now loads the API key securely using Streamlit's secrets management.
# For deployment, you must set the key under 'GEMINI_API_KEY' in your secrets.toml file.
# The platform handles the key injection, ensuring it works here as well.
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    # Fallback: If the key isn't in st.secrets, check environment variables.
    # This ensures compatibility with common deployment and local testing methods.
    API_KEY = os.environ.get("GEMINI_API_KEY", "")

# The URL for the Gemini API model endpoint
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

# --- Hardcoded Normalized Bounding Boxes (0-1000) ---
# Coordinates are estimated based on a typical NRC layout and assume a CORRECTLY ORIENTED image.
NRC_BOUNDING_BOXES = {
    # Normalized coordinates [xMin, yMin, xMax, yMax] based on a 1000x1000 grid
    "NRC_State_Code": [190, 200, 290, 260],
    "NRC_Number": [300, 200, 950, 290],
    "Name": [280, 360, 950, 440],            # Handwritten Field
    "Fathers_Name": [280, 450, 950, 530],    # Handwritten Field
    "Date_of_Birth": [280, 530, 700, 610],   # Handwritten Field
    "Height": [280, 610, 550, 690],
    "Religion": [280, 690, 700, 770],
    "Blood_Type": [280, 770, 550, 850],
}


# --- Data Mapping and Schemas (Unchanged) ---
JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "Overall_Confidence_Score": {
            "type": "NUMBER",
            "description": "A score from 0.0 to 1.0 indicating the model's certainty in the entire extraction, especially for handwritten Burmese fields. Use 0.0 to 1.0 format."
        },
        "NRC_State_Code": {
            "type": "STRING",
            "description": "The State/Division code (the first 1-2 digits/characters of the NRC), e.g., '1', '7', '12', translated into **standard Latin digits (1-14)**."
        },
        "NRC_Number": {
            "type": "STRING",
            "description": "The complete NRC number, including codes and the 6-digit number, following the pattern: X/XXX(Y)######. Must use standard characters."
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
        },
        "Height": {
            "type": "STRING",
            "description": "The cardholder's height (အရပ်), in the original Burmese script and units."
        },
        "Religion": {
            "type": "STRING",
            "description": "The cardholder's religion (ကိုးကွယ်သည့်ဘာသာ), in the original Burmese script."
        },
        "Blood_Type": {
            "type": "STRING",
            "description": "The cardholder's blood type (သွေးအမျိုးအစား), including the Rh factor, if available."
        }
    },
    "required": ["Overall_Confidence_Score", "NRC_State_Code", "NRC_Number", "Name", "Fathers_Name", "Date_of_Birth"] 
}

SYSTEM_INSTRUCTION = (
    "You are an expert Optical Character Recognition (OCR) and Intelligent Document "
    "Processing (IDP) system specialized in reading Myanmar National Registration Card (NRC) documents. "
    "Your primary task is to **meticulously and accurately** extract the required fields, prioritizing the **precise recognition "
    "of handwritten Burmese script**, even when the image quality is imperfect. "
    "For handwritten fields, you must ignore the underlying pre-printed text/patterns and focus solely on the script written with a pen. "
    "All numerical values (like the State Code) must be output as standard Latin digits (0-9). "
    "Crucially, you must provide an Overall_Confidence_Score (0.0 to 1.0) based on the image quality and legibility of the handwritten fields. "
    "Output the results ONLY as a JSON object conforming to the provided schema."
)

# --- NEW: Rotation Correction Function ---
def rotate_image_from_exif(img: Image.Image) -> Image.Image:
    """
    Reads EXIF data to automatically correct image orientation.
    """
    try:
        exif = img._getexif()
        if exif is not None:
            # Find the Orientation tag ID (usually 274)
            orientation_tag = next((k for k, v in ExifTags.TAGS.items() if v == 'Orientation'), None)
            
            orientation = exif.get(orientation_tag)
            
            # Apply corresponding rotation
            if orientation == 3:
                img = img.rotate(180, expand=True)
                st.info("Auto-rotated image 180° based on EXIF data.")
            elif orientation == 6:
                img = img.rotate(-90, expand=True) # Rotate 90 degrees CCW
                st.info("Auto-rotated image -90° based on EXIF data.")
            elif orientation == 8:
                img = img.rotate(90, expand=True) # Rotate 90 degrees CW
                st.info("Auto-rotated image 90° based on EXIF data.")
    except Exception:
        # Ignore if EXIF data is missing or corrupted
        pass
    
    return img

def image_to_bytes(img: Image.Image) -> bytes:
    """Converts a PIL Image object back to JPEG bytes."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


# --- Image Pre-Processing Pipeline ---

def process_image(image_bytes: bytes, rotation_angle: int = 0) -> bytes:
    """
    Applies auto-rotation, manual rotation, sharpening, and contrast enhancement.
    """
    try:
        # Load the image
        img = Image.open(io.BytesIO(image_bytes))

        # 1. Auto-Rotate from EXIF data (Done first to set correct baseline)
        img = rotate_image_from_exif(img)
        
        # 2. Apply Manual Rotation (if triggered by user buttons)
        if rotation_angle != 0:
            img = img.rotate(rotation_angle, expand=True)
            st.info(f"Manually rotated image by {rotation_angle}°.")

        # 3. Sharpen the image (Factor of 2.0)
        sharpen_filter = ImageEnhance.Sharpness(img)
        img_sharpened = sharpen_filter.enhance(2.0)
        
        # 4. Increase contrast (Factor of 1.5)
        contrast_filter = ImageEnhance.Contrast(img_sharpened)
        img_final = contrast_filter.enhance(1.5)
        
        # 5. Convert back to bytes and store the PIL object for later use
        final_bytes = image_to_bytes(img_final)

        # Store the manually rotated PIL image in session state for re-runs
        st.session_state['current_image_bytes'] = image_to_bytes(img)
        
        return final_bytes
    except Exception as e:
        st.warning(f"Image processing pipeline failed: {e}. Using original image bytes.")
        return image_bytes

# --- Validation and Utility Functions (Unchanged) ---

def is_valid_date(burmese_date_str: str) -> bool:
    """Tries to parse a date from the Burmese string and validates it."""
    today = date.today()
    import re
    year_match = re.search(r'(\d{4})', burmese_date_str)
    
    if year_match:
        try:
            year = int(year_match.group(1))
            if year > today.year: return False
            if today.year - year > 200: return False
            return True
        except ValueError:
            return False
    return True 

def validate_nrc_data(data: Dict[str, Any]) -> List[str]:
    """Applies the specified business rules to the extracted NRC data."""
    warnings = []
    
    # 1. NRC State/Division Code should be 1 to 14
    nrc_state_code = str(data.get('NRC_State_Code', '')).strip()
    if nrc_state_code:
        try:
            code = int(nrc_state_code)
            if not (1 <= code <= 14):
                warnings.append(f"State Code '{nrc_state_code}' is outside the valid range (1-14).")
        except ValueError:
            warnings.append(f"State Code '{nrc_state_code}' is not a valid number (1-14).")
    else: warnings.append("NRC State Code is missing.")
        
    # 2. Citizen Number should always include the pattern (X)######
    nrc_number_full = data.get('NRC_Number', '').strip()
    import re
    # Simplified regex check for a classification and 6 digits
    citizen_number_match = re.search(r'\([CNA]\)\d{6}$', nrc_number_full)
    if not citizen_number_match:
        warnings.append(f"NRC Number '{nrc_number_full}' may be missing the classification code or the 6-digit citizen number.")
        
    # 3. Date of birth should be valid date (checks 4-digit year)
    dob_burmese = data.get('Date_of_Birth', '').strip()
    if dob_burmese and not is_valid_date(dob_burmese):
        warnings.append(f"Date of Birth '{dob_burmese}' appears invalid (e.g., future date or over 200 years old).")
        
    return warnings

def calculate_accuracy_score(original_data: Dict[str, Any], corrected_data: Dict[str, Any]) -> float:
    """
    Calculates a field-level string similarity score between the model's output and the human-corrected output.
    Uses sequence matcher ratio (a measure of string similarity).
    """
    fields_to_compare = [
        "NRC_State_Code", "NRC_Number", "Name", "Fathers_Name", 
        "Date_of_Birth", "Height", "Religion", "Blood_Type"
    ]
    
    total_score = 0.0
    
    for field in fields_to_compare:
        original_value = str(original_data.get(field, "")).strip()
        corrected_value = str(corrected_data.get(field, "")).strip()
        
        if original_value or corrected_value:
            similarity_ratio = difflib.SequenceMatcher(None, original_value, corrected_value).ratio()
            total_score += similarity_ratio
        else:
            total_score += 1.0 # Treat empty/empty as 100% accurate for that optional field
            
    accuracy = total_score / len(fields_to_compare)
    return accuracy

# --- Function to call the Gemini API ---
@st.cache_data(show_spinner=False)
def extract_nrc_data(enhanced_image_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Calls the Gemini API to extract structured data from the ENHANCED image bytes,
    using segmented extraction based on normalized coordinates.
    """
    if not API_KEY or not API_URL:
        st.error("API Key or URL is missing. Please configure them.")
        return None
        
    st.info("Sending **segmented, oriented, and enhanced** document to Gemini API for precise extraction. This may take a moment...")
    
    base64_image = base64.b64encode(enhanced_image_bytes).decode('utf-8')
    mime_type = "image/jpeg" 
    
    # --- CONSTRUCT SEGMENTED USER QUERY ---
    prompt_segments = [
        "Analyze the provided Myanmar NRC document image. Extract data for the fields listed below.",
        "The coordinates are normalized to a 1000x1000 grid [xMin, yMin, xMax, yMax].",
        "For each field, only read the text within the specified bounding box. Ignore all other content and background.\n"
    ]
    
    # Add a specific instruction for each field using the bounding box
    for field_name, coords in NRC_BOUNDING_BOXES.items():
        # Convert 0-1000 coordinates to 0.0-1.0 format for better AI interpretation
        normalized_coords = [c / 1000.0 for c in coords]
        
        instruction = f"- Field: **{field_name}** at: **[{normalized_coords[0]:.3f}, {normalized_coords[1]:.3f}, {normalized_coords[2]:.3f}, {normalized_coords[3]:.3f}]**."
        
        if field_name in ["Name", "Fathers_Name", "Date_of_Birth"]:
            instruction += " **MUST be extracted from handwritten Burmese script.**"
        elif field_name == "NRC_State_Code":
            instruction += " Output as a standard Latin digit (1-14)."
        
        prompt_segments.append(instruction)

    # Final query assembly
    user_query = "\n".join(prompt_segments)
    user_query += "\n\nReturn the output as a single JSON object."
    
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
            API_URL, 
            params={'key': API_KEY},
            headers={'Content-Type': 'application/json'}, 
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        
        # Safely extract the JSON string
        json_string = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
        
        if json_string:
            extracted_data = json.loads(json_string)
            
            st.session_state['original_data'] = extracted_data.copy()
            
            # Run Validation
            warnings = validate_nrc_data(extracted_data)
            extracted_data['validation_warnings'] = warnings
            
            return extracted_data
        else:
            error_detail = result.get('candidates', [{}])[0].get('finishReason', 'No content generated.')
            st.error(f"Error: Could not extract structured JSON. Finish Reason: {error_detail}")
            st.code(result, language='json')
            return None

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


# --- Custom Logic to Handle Text Input Updates (Unchanged) ---

def update_data_from_fields(fields_data: Dict[str, str]):
    """
    Gathers data from Streamlit text inputs, re-validates, calculates accuracy, and updates the state.
    """
    try:
        # 1. Construct the updated data structure
        updated_data = {
            "Overall_Confidence_Score": st.session_state['extracted_data'].get('Overall_Confidence_Score', 0.0),
            "NRC_State_Code": fields_data.get('NRC_State_Code', '').strip(),
            "NRC_Number": fields_data.get('NRC_Number', '').strip(),
            "Name": fields_data.get('Name', '').strip(),
            "Fathers_Name": fields_data.get('Fathers_Name', '').strip(),
            "Date_of_Birth": fields_data.get('Date_of_Birth', '').strip(),
            "Height": fields_data.get('Height', '').strip(),
            "Religion": fields_data.get('Religion', '').strip(),
            "Blood_Type": fields_data.get('Blood_Type', '').strip(),
        }

        # Check for REQUIRED fields after cleaning
        required_fields_check = all(updated_data.get(field) for field in JSON_SCHEMA['required'] if field != 'Overall_Confidence_Score')
        if not required_fields_check:
             st.error("Error: All required fields must have a value.")
             return
        
        # 2. Re-validate the human-corrected data
        updated_data['validation_warnings'] = validate_nrc_data(updated_data)

        # 3. Calculate Accuracy Score
        original_data = st.session_state.get('original_data', {})
        accuracy = calculate_accuracy_score(original_data, updated_data)
        st.session_state['accuracy_score'] = accuracy
        
        # 4. Update session state
        st.session_state['extracted_data'] = updated_data
        st.success(f"✅ Data successfully updated and re-validated. Accuracy Score: {accuracy * 100:.2f}%")
    except Exception as e:
        st.error(f"An unexpected error occurred during update: {e}")

# --- Rotation Action Functions ---
def rotate_uploaded_image(angle: int):
    """
    Applies manual rotation to the original uploaded image bytes and stores them.
    """
    if 'uploaded_file_bytes' not in st.session_state or st.session_state['uploaded_file_bytes'] is None:
        st.error("Please upload an image first.")
        return
        
    try:
        # Load the image from the latest bytes (which might already be rotated)
        img_bytes = st.session_state.get('current_image_bytes') or st.session_state['uploaded_file_bytes']
        img = Image.open(io.BytesIO(img_bytes))
        
        # Apply rotation
        img_rotated = img.rotate(angle, expand=True)
        
        # Save the new bytes back to session state for re-runs
        st.session_state['uploaded_file_bytes'] = image_to_bytes(img_rotated)
        st.session_state['current_image_bytes'] = image_to_bytes(img_rotated)

        # Clear old extraction results
        st.session_state['extracted_data'] = None 
        st.session_state['accuracy_score'] = None
        st.session_state['enhanced_image_bytes'] = None 
        
        st.rerun() # Rerun to update the display
    except Exception as e:
        st.error(f"Error during manual rotation: {e}")


# --- Streamlit App UI ---

def main():
    st.set_page_config(page_title="NRC Document Data Extractor V4", layout="centered")

    st.title("🇲🇲 NRC Document Data Extractor V4")
    st.subheader("Step 3: Robustness (Auto-Rotation, Segmentation, and Enhancement)")
    st.markdown("This version automatically corrects the image orientation via **EXIF data** to ensure the bounding boxes are accurate. Use the rotation buttons if the image is still crooked.")

    # Initialize session state for data storage
    if 'extracted_data' not in st.session_state: st.session_state['extracted_data'] = None
    if 'original_data' not in st.session_state: st.session_state['original_data'] = None
    if 'accuracy_score' not in st.session_state: st.session_state['accuracy_score'] = None
    if 'enhanced_image_bytes' not in st.session_state: st.session_state['enhanced_image_bytes'] = None
    if 'uploaded_file_bytes' not in st.session_state: st.session_state['uploaded_file_bytes'] = None
    if 'current_image_bytes' not in st.session_state: st.session_state['current_image_bytes'] = None


    # --- INPUT SELECTION ---
    tab1, tab2 = st.tabs(["🖼️ Upload Image", "📸 Take Photo"])

    uploaded_file = None
    camera_image = None
    selected_source = None

    with tab1:
        uploaded_file = st.file_uploader("Upload NRC Image (PNG, JPG)", type=["png", "jpg", "jpeg"])
    with tab2:
        camera_image = st.camera_input("Snap a photo of the NRC Document")
    
    # Determine the latest uploaded source
    if uploaded_file is not None:
        selected_source = uploaded_file
        # Initialize or update session state with new upload
        if st.session_state.get('uploaded_file_name') != uploaded_file.name:
            st.session_state['uploaded_file_bytes'] = uploaded_file.getvalue()
            st.session_state['uploaded_file_name'] = uploaded_file.name
            st.session_state['current_image_bytes'] = uploaded_file.getvalue()
    elif camera_image is not None:
        selected_source = camera_image
        # Always treat camera image as a new input
        st.session_state['uploaded_file_bytes'] = camera_image.getvalue()
        st.session_state['current_image_bytes'] = camera_image.getvalue()
        
    
    if st.session_state.get('uploaded_file_bytes'):
        image_bytes_to_display = st.session_state['current_image_bytes']
        
        # --- Rotation Controls ---
        st.subheader("0. Correct Orientation (If needed)")
        col_rot1, col_rot2, col_rot3, col_rot4 = st.columns([1, 1, 1, 3])
        with col_rot1:
            st.button("Rotate ↺ -90°", on_click=rotate_uploaded_image, args=(90,), help="Rotate Counter-Clockwise 90°")
        with col_rot2:
            st.button("Rotate ↻ +90°", on_click=rotate_uploaded_image, args=(-90,), help="Rotate Clockwise 90°")
        with col_rot3:
            st.button("Rotate 180°", on_click=rotate_uploaded_image, args=(180,), help="Rotate 180°")
        with col_rot4:
            st.info("The image below shows the *current* orientation.")

        # --- Image Display Tabs ---
        img_tab1, img_tab2 = st.tabs(["Current Oriented Image", "AI Processed Image (Enhanced)"])
        
        with img_tab1:
            # Display the current, oriented but unenhanced image
            st.image(image_bytes_to_display, caption='Current Input Document (Auto-rotated via EXIF if metadata was present)', use_column_width=True)

        with img_tab2:
            # Display the final, enhanced image that goes to the AI
            if 'enhanced_image_bytes' in st.session_state and st.session_state['enhanced_image_bytes'] is not None:
                st.image(st.session_state['enhanced_image_bytes'], caption='Sharpened & Contrast Enhanced Image Sent to AI', use_column_width=True)
            else:
                st.info("The AI processed image preview will appear here after extraction.")

        st.divider()

        # Button to trigger extraction
        if st.button("1. Extract & Validate Data (Using Segmented Prompt)", type="primary"):
            # The extraction uses the currently oriented image bytes
            
            # 1. Clear previous results (except for the current image bytes)
            st.session_state['extracted_data'] = None 
            st.session_state['accuracy_score'] = None
            
            with st.spinner("Processing image (Auto-Rotation, Enhancement) and calling Gemini..."):
                # 2. Run the image through the pipeline for enhancement
                enhanced_bytes = process_image(st.session_state['uploaded_file_bytes'])
                st.session_state['enhanced_image_bytes'] = enhanced_bytes
                
                # 3. Perform the extraction
                extracted_data = extract_nrc_data(enhanced_bytes)
            
            if extracted_data:
                st.session_state['extracted_data'] = extracted_data
                st.session_state['accuracy_score'] = 1.0 # Reset accuracy for new run
                st.rerun() 
                
        # --- Display Results, Validation, and Correction ---
        if st.session_state['extracted_data'] is not None:
            data = st.session_state['extracted_data']
            warnings = data.get('validation_warnings', [])
            confidence = data.get('Overall_Confidence_Score')
            accuracy = st.session_state.get('accuracy_score', 1.0)
            
            # --- Performance Metrics Section ---
            st.header("✨ Performance Metrics")
            col_conf, col_acc = st.columns(2)
            
            with col_conf:
                st.metric(
                    label="Model Confidence",
                    value=f"{confidence * 100:.0f}%" if confidence is not None else "N/A",
                    delta="AI's certainty of its initial output."
                )
            with col_acc:
                st.metric(
                    label="OCR Field Accuracy",
                    value=f"{accuracy * 100:.2f}%",
                    delta="Similarity to the human-corrected output."
                )

            # 2. Validation Warnings
            st.subheader("⚠️ Validation Warnings")
            if warnings:
                st.warning("The extracted data has the following potential errors:")
                for warning in warnings:
                    st.markdown(f"- **{warning}**")
            else:
                st.success("Data passed all preliminary validation checks.")
            
            st.markdown("---")

            # 3. Human-in-the-Loop Correction (Text Inputs)
            st.subheader("2. Human-in-the-Loop Correction (Text Fields)")
            st.markdown("Review and correct any errors below. **Any change will be used to calculate the model's accuracy.**")
            
            # --- Text Input Fields for Correction ---
            with st.form("correction_form"):
                
                current_data = st.session_state['extracted_data']
                
                # --- NRC Fields ---
                col1, col2 = st.columns(2)
                with col1:
                    NRC_State_Code = st.text_input(
                        "NRC State Code (1-14)",
                        value=current_data.get('NRC_State_Code', ''),
                        key='NRC_State_Code_input',
                        help="e.g., 1, 7, 12 (Must be a number 1-14)"
                    )
                with col2:
                    NRC_Number = st.text_input(
                        "NRC Full Number (အမှတ်)",
                        value=current_data.get('NRC_Number', ''),
                        key='NRC_Number_input',
                        help="e.g., 1/KaMaNa(N)123456"
                    )

                # --- Personal Details ---
                Name = st.text_input(
                    "Name (အမည်)",
                    value=current_data.get('Name', ''),
                    key='Name_input',
                    help="Enter the cardholder's name in original Burmese script."
                )
                Fathers_Name = st.text_input(
                    "Father's Name (အဘအမည်)",
                    value=current_data.get('Fathers_Name', ''),
                    key='Fathers_Name_input',
                    help="Enter the father's name in original Burmese script."
                )
                Date_of_Birth = st.text_input(
                    "Date of Birth (မွေးသက္ကရာဇ်)",
                    value=current_data.get('Date_of_Birth', ''),
                    key='Date_of_Birth_input',
                    help="Enter the date in original Burmese script (e.g., date, month, year)."
                )

                # --- Optional Fields ---
                col3, col4, col5 = st.columns(3)
                with col3:
                    Height = st.text_input("Height (အရပ်)", value=current_data.get('Height', ''), key='Height_input')
                with col4:
                    Religion = st.text_input("Religion (ကိုးကွယ်သည့်ဘာသာ)", value=current_data.get('Religion', ''), key='Religion_input')
                with col5:
                    Blood_Type = st.text_input("Blood Type (သွေးအမျိုးအစား)", value=current_data.get('Blood_Type', ''), key='Blood_Type_input')

                # Button to submit corrections
                submitted = st.form_submit_button("Update and Re-Validate Extracted Data", type="secondary")
                
                if submitted:
                    fields_data = {
                        "NRC_State_Code": NRC_State_Code,
                        "NRC_Number": NRC_Number,
                        "Name": Name,
                        "Fathers_Name": Fathers_Name,
                        "Date_of_Birth": Date_of_Birth,
                        "Height": Height,
                        "Religion": Religion,
                        "Blood_Type": Blood_Type,
                    }
                    update_data_from_fields(fields_data)
                    st.rerun()
            
            st.markdown("---")

            # 4. Final Data Display and Download
            final_data_view = [
                ("Confidence Score", f"{data.get('Overall_Confidence_Score', 0.0) * 100:.0f}%"), 
                ("Accuracy Score", f"{accuracy * 100:.2f}%"), 
                ("NRC State Code", data.get("NRC_State_Code", "N/A")),
                ("NRC Number (အမှတ်)", data.get("NRC_Number", "N/A")),
                ("Name (အမည်)", data.get("Name", "N/A")),
                ("Father's Name (အဘအမည်)", data.get("Fathers_Name", "N/A")),
                ("Date of Birth (မွေးသက္ကရာဇ်)", data.get("Date_of_Birth", "N/A")),
                ("Height (အရပ်)", data.get("Height", "N/A")),
                ("Religion (ကိုးကွယ်သည့်ဘာသာ)", data.get("Religion", "N/A")),
                ("Blood Type (သွေးအမျိုးအစား)", data.get("Blood_Type", "N/A"))
            ]
            st.subheader("3. Final Corrected Data")
            st.table(final_data_view)
            
            # Download Button (Data remains the same structure)
            download_data = data.copy()
            download_data.pop('validation_warnings', None)
            download_data['OCR_Accuracy_Score'] = accuracy 
            
            json_output = json.dumps(download_data, indent=2)
            
            header = list(download_data.keys())
            values = [str(download_data[k]) for k in header]
            csv_output = ",".join(header) + "\n" + ",".join(values)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="⬇️ Download Final JSON",
                    data=json_output,
                    file_name="nrc_data_corrected.json",
                    mime="application/json",
                )
            with col_dl2:
                st.download_button(
                    label="⬇️ Download Final CSV",
                    data=csv_output,
                    file_name="nrc_data_corrected.csv",
                    mime="text/csv",
                )


    else:
        st.info("Please upload an image file or take a photo and click '1. Extract & Validate Data' to begin. Use the rotation tools if your image is not upright.")

if __name__ == "__main__":
    main()