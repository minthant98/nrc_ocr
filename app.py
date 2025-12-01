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

# --- Data Mapping and Schemas (UPDATED: Core Fields Only) ---
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

# --- SYSTEM INSTRUCTION (UPDATED: Focus on Core Fields) ---
SYSTEM_INSTRUCTION = (
    "You are an expert Optical Character Recognition (OCR) and Intelligent Document "
    "Processing (IDP) system specialized in reading Myanmar National Registration Card (NRC) documents. "
    "Your primary task is to **meticulously and accurately** extract ONLY the core identity fields: the full NRC number breakdown, "
    "Name, Father's Name, and Date of Birth. Prioritize the **precise recognition of handwritten Burmese script** "
    "for fields like Name, Fathers_Name, Date_of_Birth, and NRC_township. "
    "The NRC number must be broken down into four components: NRC_state_division (1-14, Latin digits), NRC_township (Burmese script), "
    "NRC_sth (e.g., (N), Latin letters), and NRC_no (6 digits, Latin digits). "
    "All numerical values must be output as standard Latin digits (0-9). "
    "Crucially, you must provide an Overall_Confidence_Score (0.0 to 1.0) based on the image quality and legibility. "
    "Output the results ONLY as a JSON object conforming to the provided schema."
)


# --- Image Pre-Processing Pipeline (Unchanged) ---

def rotate_image_from_exif(img: Image.Image) -> Image.Image:
    """
    Reads EXIF data to automatically correct image orientation.
    """
    try:
        exif = img._getexif()
        if exif is not None:
            orientation_tag = next((k for k, v in ExifTags.TAGS.items() if v == 'Orientation'), None)
            orientation = exif.get(orientation_tag)
            
            if orientation == 3:
                img = img.rotate(180, expand=True)
                st.info("Auto-rotated image 180° based on EXIF data.")
            elif orientation == 6:
                img = img.rotate(-90, expand=True)
                st.info("Auto-rotated image -90° based on EXIF data.")
            elif orientation == 8:
                img = img.rotate(90, expand=True)
                st.info("Auto-rotated image 90° based on EXIF data.")
    except Exception:
        pass
    
    return img

def image_to_bytes(img: Image.Image) -> bytes:
    """Converts a PIL Image object back to JPEG bytes."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()

def process_image(image_bytes: bytes, rotation_angle: int = 0) -> bytes:
    """
    Applies auto-rotation, manual rotation, sharpening, and contrast enhancement.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # 1. Auto-Rotate from EXIF data
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
    
    # 1. Validate NRC State/Division Code
    nrc_state_code = str(data.get('NRC_state_division', '')).strip()
    if nrc_state_code:
        try:
            code = int(nrc_state_code)
            if not (1 <= code <= 14):
                warnings.append(f"State/Division Code '{nrc_state_code}' is outside the valid range (1-14).")
        except ValueError:
            warnings.append(f"State/Division Code '{nrc_state_code}' is not a valid number (1-14).")
    else: warnings.append("NRC State/Division Code is missing.")
    
    # 2. Validate NRC 6-digit Number
    nrc_no = str(data.get('NRC_no', '')).strip()
    if not (nrc_no.isdigit() and len(nrc_no) == 6):
        warnings.append(f"NRC 6-digit number '{nrc_no}' is invalid. It must contain exactly 6 Latin digits.")
    
    # 3. Validate NRC Status (sth)
    nrc_sth = str(data.get('NRC_sth', '')).strip()
    if nrc_sth and not (nrc_sth.startswith('(') and nrc_sth.endswith(')') and len(nrc_sth) >= 3):
        warnings.append(f"NRC Classification Code ('NRC_sth') '{nrc_sth}' should be formatted as (C), (N), or (A) and include parentheses.")

    # 4. Validate Date of Birth
    dob_burmese = data.get('Date_of_Birth', '').strip()
    if dob_burmese and not is_valid_date(dob_burmese):
        warnings.append(f"Date of Birth '{dob_burmese}' appears invalid (e.g., future date or over 200 years old).")
        
    return warnings

# --- Accuracy Score (UPDATED: Core Fields Only) ---
def calculate_accuracy_score(original_data: Dict[str, Any], corrected_data: Dict[str, Any]) -> float:
    """
    Calculates a field-level string similarity score between the model's output and the human-corrected output.
    Focuses only on the core fields: NRC components, Name, Fathers_Name, and Date_of_Birth.
    """
    fields_to_compare = [
        "NRC_state_division", "NRC_township", "NRC_sth", "NRC_no", 
        "Name", "Fathers_Name", "Date_of_Birth"
    ]
    total_score = 0.0
    
    for field in fields_to_compare:
        original_value = str(original_data.get(field, "")).strip()
        corrected_value = str(corrected_data.get(field, "")).strip()
        
        if original_value or corrected_value:
            similarity_ratio = difflib.SequenceMatcher(None, original_value, corrected_value).ratio()
            total_score += similarity_ratio
        else:
            total_score += 1.0 
            
    accuracy = total_score / len(fields_to_compare)
    return accuracy

def update_data_from_fields(fields_data: Dict[str, str]):
    """
    Updates the session state with human-corrected data from the form, re-validates, 
    and recalculates the accuracy score.
    """
    # 1. Update the extracted_data in session state with corrected fields
    # Use .copy() to ensure we are modifying a mutable copy
    updated_data = st.session_state['extracted_data'].copy() 
    for key, value in fields_data.items():
        updated_data[key] = value

    st.session_state['extracted_data'] = updated_data
    
    # 2. Re-validate the corrected data
    updated_data['validation_warnings'] = validate_nrc_data(updated_data)

    # 3. Recalculate accuracy against the original model output
    original_data = st.session_state['original_data']
    accuracy = calculate_accuracy_score(original_data, updated_data)
    st.session_state['accuracy_score'] = accuracy
    
    st.success("Data successfully updated, re-validated, and accuracy score recalculated.")

# --- Function to call the Gemini API ---
@st.cache_data(show_spinner=False)
def extract_nrc_data(enhanced_image_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Calls the Gemini API to extract structured data from the ENHANCED image bytes,
    relying on holistic understanding of the document structure.
    """
    if not API_KEY or not API_URL:
        st.error("API Key or URL is missing. Please configure them.")
        return None
        
    st.info("Sending **oriented and enhanced** document to Gemini API for extraction. The model will use structural context to find the fields.")
    
    base64_image = base64.b64encode(enhanced_image_bytes).decode('utf-8')
    mime_type = "image/jpeg" 
    
    # --- CONSTRUCT HOLISTIC USER QUERY (Emphasizing core fields) ---
    user_query = (
        "Analyze the provided Myanmar NRC document image. Extract the values for ONLY the core identity fields: "
        "the four NRC components, Name, Father's Name, and Date of Birth. "
        "Crucially, split the NRC number (X/XXX(Y)######) into its four requested components: "
        "1. NRC_state_division (X, Latin digits 1-14) "
        "2. NRC_township (XXX, Burmese words between '/' and '(', in original Burmese script) "
        "3. NRC_sth ((Y), the classification code including parentheses, e.g., (N), (C), or (A)) "
        "4. NRC_no (######, the 6-digit number, Latin digits) "
        "Ensure Name, Father's Name, and Date of Birth are copied exactly as written in the original handwritten Burmese script. "
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
            API_URL, 
            params={'key': API_KEY},
            headers={'Content-Type': 'application/json'}, 
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        
        json_string = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
        
        if json_string:
            extracted_data = json.loads(json_string)
            st.session_state['original_data'] = extracted_data.copy()
            warnings = validate_nrc_data(extracted_data)
            extracted_data['validation_warnings'] = warnings
            return extracted_data
        else:
            error_detail = result.get('candidates', [{}])[0].get('finishReason', 'No content generated.')
            st.error(f"Error: Could not extract structured JSON. Finish Reason: {error_detail}")
            return None

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


# --- Rotation Action Functions (Unchanged) ---
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


# --- Streamlit App UI (UPDATED FOR CORE FIELDS) ---

def main():
    st.set_page_config(page_title="NRC Document Data Extractor V7", layout="centered")

    st.title("MM NRC Document Data Extractor V7")
    st.subheader("Focused Extraction: Core Identity Fields Only")
    st.markdown("This version focuses solely on the **NRC number breakdown (4 parts), Name, Father's Name, and Date of Birth** for maximum accuracy on critical data.")

    # Initialize session state for data storage
    if 'extracted_data' not in st.session_state: st.session_state['extracted_data'] = None
    if 'original_data' not in st.session_state: st.session_state['original_data'] = None
    if 'accuracy_score' not in st.session_state: st.session_state['accuracy_score'] = None
    if 'enhanced_image_bytes' not in st.session_state: st.session_state['enhanced_image_bytes'] = None
    if 'uploaded_file_bytes' not in st.session_state: st.session_state['uploaded_file_bytes'] = None
    if 'current_image_bytes' not in st.session_state: st.session_state['current_image_bytes'] = None
    if 'uploaded_file_name' not in st.session_state: st.session_state['uploaded_file_name'] = None


    # --- INPUT SELECTION ---
    tab1, tab2 = st.tabs(["🖼️ Upload Image", "📸 Take Photo"])

    uploaded_file = None
    
    # Process file/camera input using local variables 
    with tab1:
        uploaded_file_widget = st.file_uploader("Upload Myanmar NRC Image", type=["png", "jpg", "jpeg"])
    with tab2:
        camera_image_widget = st.camera_input("Take a Photo of NRC")

    if uploaded_file_widget:
        uploaded_file = uploaded_file_widget
    elif camera_image_widget:
        uploaded_file = camera_image_widget
        
    
    if uploaded_file:
        file_name_or_default = getattr(uploaded_file, 'name', 'camera_image_file')

        # Only update session state bytes if a new file/photo is detected
        if st.session_state['uploaded_file_name'] != file_name_or_default:
            st.session_state['uploaded_file_bytes'] = uploaded_file.getvalue()
            st.session_state['uploaded_file_name'] = file_name_or_default
            st.session_state['current_image_bytes'] = uploaded_file.getvalue() # Stores the potentially initial (or rotated) image

    
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
            st.info("The image below shows the *current* orientation (after auto-EXIF correction).")

        # --- Image Display Tabs ---
        img_tab1, img_tab2 = st.tabs(["Current Oriented Image", "AI Processed Image (Enhanced)"])
        
        with img_tab1:
            st.image(image_bytes_to_display, caption='Current Input Document', use_column_width=True)

        with img_tab2:
            if 'enhanced_image_bytes' in st.session_state and st.session_state['enhanced_image_bytes'] is not None:
                st.image(st.session_state['enhanced_image_bytes'], caption='Sharpened & Contrast Enhanced Image Sent to AI', use_column_width=True)
            else:
                st.info("The AI processed image preview will appear here after extraction.")

        st.divider()

        # Button to trigger extraction
        if st.button("1. Extract & Validate Data", type="primary"):
            
            # 1. Clear previous results
            st.session_state['extracted_data'] = None 
            st.session_state['accuracy_score'] = None
            
            with st.spinner("Processing image (Auto-Orientation, Enhancement) and calling Gemini..."):
                # 2. Run the image through the pipeline for enhancement and orientation
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
            st.subheader("⚠️ 2. Validation Warnings")
            if warnings:
                st.warning("The extracted data has the following potential errors:")
                for warning in warnings:
                    st.markdown(f"- **{warning}**")
            else:
                st.success("Data passed all preliminary validation checks.")
            
            st.markdown("---")

            # 3. Final Data Display and Download (MOVED UP)
            final_data_view = [
                ("Confidence Score", f"{data.get('Overall_Confidence_Score', 0.0) * 100:.0f}%"), 
                ("Accuracy Score", f"{accuracy * 100:.2f}%"), 
                ("NRC State/Division (X)", data.get("NRC_state_division", "N/A")),
                ("NRC Township (XXX)", data.get("NRC_township", "N/A")),
                ("NRC Classification ((Y))", data.get("NRC_sth", "N/A")),
                ("NRC 6-Digit No (######)", data.get("NRC_no", "N/A")),
                ("Name (အမည်)", data.get("Name", "N/A")),
                ("Father's Name (အဘအမည်)", data.get("Fathers_Name", "N/A")),
                ("Date of Birth (မွေးသက္ကရာဇ်)", data.get("Date_of_Birth", "N/A")),
            ]
            st.subheader("✅ 3. Current Extracted Data Snapshot")
            st.table(final_data_view)

            # Download Button 
            download_data = data.copy()
            download_data.pop('validation_warnings', None)
            download_data['OCR_Accuracy_Score'] = accuracy 
            
            json_output = json.dumps(download_data, indent=2)
            
            # Create CSV headers and values from the data keys
            header = list(download_data.keys())
            values = [str(download_data[k]) for k in header]
            csv_output = ",".join(header) + "\n" + ",".join(values)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="⬇️ Download Final JSON",
                    data=json_output,
                    file_name="nrc_data_corrected_v7.json",
                    mime="application/json",
                )
            with col_dl2:
                st.download_button(
                    label="⬇️ Download Final CSV",
                    data=csv_output,
                    file_name="nrc_data_corrected_v7.csv",
                    mime="text/csv",
                )

            st.markdown("---")

            # 4. Human-in-the-Loop Correction (Text Fields) (MOVED DOWN & SIMPLIFIED)
            st.subheader("✍️ 4. Correct Data & Recalculate Accuracy")
            st.markdown("Review and correct any errors below. **Only core fields are shown.** Any change will be used to calculate the model's accuracy.")
            
            # --- Text Input Fields for Correction ---
            with st.form("correction_form"):
                
                current_data = st.session_state['extracted_data']
                
                # --- Granular NRC Fields ---
                st.markdown("### NRC Components (X/XXX(Y)######)")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    NRC_state_division = st.text_input(
                        "State/Division Code (X)",
                        value=current_data.get('NRC_state_division', ''),
                        key='NRC_state_division_input',
                        help="Must be a Latin digit 1-14."
                    )
                with col2:
                    NRC_township = st.text_input(
                        "Township Code (XXX)",
                        value=current_data.get('NRC_township', ''),
                        key='NRC_township_input',
                        help="Burmese script between '/' and '('."
                    )
                with col3:
                    NRC_sth = st.text_input(
                        "Classification Code ((Y))",
                        value=current_data.get('NRC_sth', ''),
                        key='NRC_sth_input',
                        help="e.g., (N), (C), or (A) including parentheses."
                    )
                with col4:
                    NRC_no = st.text_input(
                        "6-Digit Number (######)",
                        value=current_data.get('NRC_no', ''),
                        key='NRC_no_input',
                        help="Must be exactly 6 Latin digits."
                    )

                st.markdown("---")

                # --- Personal Details ---
                Name = st.text_input(
                    "Name (အမည်)",
                    value=current_data.get('Name', ''),
                    key='Name_input',
                    help="Enter the cardholder's name in original handwritten Burmese script."
                )
                Fathers_Name = st.text_input(
                    "Father's Name (အဘအမည်)",
                    value=current_data.get('Fathers_Name', ''),
                    key='Fathers_Name_input',
                    help="Enter the father's name in original handwritten Burmese script."
                )
                Date_of_Birth = st.text_input(
                    "Date of Birth (မွေးသက္ကရာဇ်)",
                    value=current_data.get('Date_of_Birth', ''),
                    key='Date_of_Birth_input',
                    help="Enter the date in original handwritten Burmese script (e.g., date, month, year)."
                )

                # Button to submit corrections
                submitted = st.form_submit_button("Update and Re-Validate Extracted Data", type="secondary")
                
                if submitted:
                    # Only include the fields present in the form
                    fields_data = {
                        "NRC_state_division": NRC_state_division,
                        "NRC_township": NRC_township,
                        "NRC_sth": NRC_sth,
                        "NRC_no": NRC_no,
                        "Name": Name,
                        "Fathers_Name": Fathers_Name,
                        "Date_of_Birth": Date_of_Birth,
                    }
                    update_data_from_fields(fields_data)
                    st.rerun()
            
            st.markdown("---")

    else:
        st.info("Please upload an image file or take a photo and click '1. Extract & Validate Data' to begin. Use the rotation tools if your image is not upright.")

if __name__ == "__main__":
    main()