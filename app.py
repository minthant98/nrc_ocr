import streamlit as st
import requests
import base64
import json
import io
import os # Required for checking environment variables
from datetime import datetime, date
from typing import Optional, Dict, Any, List

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

# --- Data Mapping and Schemas ---

# 1. Mandatory Structured Output Schema (UPDATED with Confidence Score)
JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "Overall_Confidence_Score": {
            "type": "NUMBER",
            "description": "A score from 0.0 to 1.0 (or 0 to 100) indicating the model's certainty in the entire extraction, especially for handwritten Burmese fields. Use 0.0 to 1.0 format."
        },
        "NRC_State_Code": {
            "type": "STRING",
            "description": "The State/Division code (the first 1-2 digits/characters of the NRC), e.g., '1', '7', '12', translated into standard characters."
        },
        "NRC_Number": {
            "type": "STRING",
            "description": "The complete NRC number, including codes and the 6-digit number, translated into standard characters if possible."
        },
        "Name": {
            "type": "STRING",
            "description": "The cardholder's name (အမည်), in the original Burmese script."
        },
        "Fathers_Name": {
            "type": "STRING",
            "description": "The father's name (အဘအမည်), in the original Burmese script."
        },
        "Date_of_Birth": {
            "type": "STRING",
            "description": "The cardholder's date of birth (မွေးသက္ကရာဇ်), in the original Burmese script (e.g., date, month, year)."
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
    # --- REQUIRED: Added Confidence Score and NRC_State_Code ---
    "required": ["Overall_Confidence_Score", "NRC_State_Code", "NRC_Number", "Name", "Fathers_Name", "Date_of_Birth"] 
}

# 2. System Instruction (UPDATED to request confidence score)
SYSTEM_INSTRUCTION = (
    "You are an expert Optical Character Recognition (OCR) and Intelligent Document "
    "Processing (IDP) system specialized in reading Myanmar National Registration Card (NRC) documents. "
    "Your primary task is to accurately extract the required fields, prioritizing the precise recognition "
    "of *handwritten* Burmese script, even when the image quality is imperfect or the script is ambiguous. "
    "The NRC_State_Code must be the first numerical part of the NRC number, translated into standard digits (1-14). "
    "**Crucially, you must provide an Overall_Confidence_Score (0.0 to 1.0) based on the image quality and legibility of the handwritten fields.** "
    "Output the results ONLY as a JSON object conforming to the provided schema."
)

# 3. Township/State Mapping (Partial - based on the user's request example)
NRC_STATE_TOWNSHIP_MAP: Dict[str, List[str]] = {
    # Example structure for State 1 (Kachin)
    "1": ["MKT", "MYT", "HRT", "PST", "KHT", "KST", "CHT", "HPNT", "LMT", "LPT", "MST", "NSS", "PTT", "TTT", "WMT"] # Fictional/partial list for demonstration
}

# --- Validation and Utility Functions (Unchanged from previous revision) ---

def is_valid_date(burmese_date_str: str) -> bool:
    """
    Tries to parse a date from the Burmese string and validates it.
    """
    today = date.today()
    import re
    year_match = re.search(r'(\d{4})', burmese_date_str)
    
    if year_match:
        try:
            year = int(year_match.group(1))
            if year > today.year:
                return False
            if today.year - year > 200:
                return False
            return True
        except ValueError:
            return False
            
    return True 

def validate_nrc_data(data: Dict[str, Any]) -> List[str]:
    """
    Applies the specified business rules to the extracted NRC data.
    """
    warnings = []
    
    # 1. NRC State/Division Code should be 1 to 14
    nrc_state_code = data.get('NRC_State_Code', '').strip()
    if nrc_state_code:
        try:
            code = int(nrc_state_code)
            if not (1 <= code <= 14):
                warnings.append(f"State Code '{nrc_state_code}' is outside the valid range (1-14).")
        except ValueError:
            warnings.append(f"State Code '{nrc_state_code}' is not a valid number (1-14).")
    else:
         warnings.append("NRC State Code is missing.")
         
    # 2. Citizen Number should always be six digits
    nrc_number_full = data.get('NRC_Number', '').strip()
    import re
    citizen_number_match = re.search(r'(\d{6})$', nrc_number_full)
    if not citizen_number_match:
        warnings.append(f"NRC Number '{nrc_number_full}' does not end with a 6-digit citizen number.")
        
    # 3. Date of birth should be valid date
    dob_burmese = data.get('Date_of_Birth', '').strip()
    if dob_burmese and not is_valid_date(dob_burmese):
        warnings.append(f"Date of Birth '{dob_burmese}' appears invalid (e.g., future date or over 200 years old).")
        
    return warnings

# --- Function to call the Gemini API ---
@st.cache_data(show_spinner=False)
def extract_nrc_data(image_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Calls the Gemini API to extract structured data from the image bytes and validates it.
    """
    if not API_KEY or not API_URL:
        st.error("API Key or URL is missing. Please configure them.")
        return None
        
    st.info("Sending document to Gemini API for extraction. This may take a moment...")
    
    # 1. Prepare the image data (base64 encoding)
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    mime_type = "image/png" 

    # 2. Construct the API Payload (UPDATED to request confidence score)
    user_query = (
        "Analyze the provided Myanmar NRC document image. Locate and extract the following fields "
        "and their values: 1) The **Overall Confidence Score** (0.0 to 1.0), 2) The **State Code** (first numerical part of the NRC), "
        "3) The **full NRC number** (အမှတ်), 4) The cardholder's Name (အမည်), 5) The Father's Name (အဘအမည်), "
        "6) The Date of Birth (မွေးသက္ကရာဇ်), 7) Height (အရပ်), 8) Religion (ကိုးကွယ်သည့်ဘာသာ), and 9) Blood Type (သွေးအမျိုးအစား). "
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
    
    # 3. Make the API request
    try:
        response = requests.post(
            API_URL, 
            params={'key': API_KEY},
            headers={'Content-Type': 'application/json'}, 
            json=payload
        )
        response.raise_for_status()
        
        # 4. Process the response
        result = response.json()
        
        json_string = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
        
        if json_string:
            extracted_data = json.loads(json_string)
            
            # --- NEW STEP: Run Validation ---
            warnings = validate_nrc_data(extracted_data)
            extracted_data['validation_warnings'] = warnings
            
            return extracted_data
        else:
            error_detail = result.get('candidates', [{}])[0].get('finishReason', 'No content generated.')
            st.error(f"Error: Could not extract structured JSON. Finish Reason: {error_detail}")
            st.code(result, language='json')
            return None

    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error during API call: {e}. Check your API key and quota.")
        st.code(response.text, language='json')
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- Custom Logic to Handle JSON Updates (Unchanged) ---

def update_data_from_json(json_text):
    """Parses the edited JSON text, updates the state, and re-validates."""
    try:
        updated_data = json.loads(json_text)
        
        # Simple validation: Check if it's a dict and has required fields
        if not isinstance(updated_data, dict) or not all(field in updated_data for field in JSON_SCHEMA['required']):
            st.error("Error: The updated JSON must be an object containing at least the REQUIRED fields.")
            return

        # Re-validate the human-corrected data
        updated_data['validation_warnings'] = validate_nrc_data(updated_data)

        st.session_state['extracted_data'] = updated_data
        st.session_state['json_text'] = json.dumps(updated_data, indent=2)
        st.success("✅ Data successfully updated and re-validated.")
    except json.JSONDecodeError:
        st.error("Error: Invalid JSON format. Please check syntax (e.g., quotes, commas, brackets).")
    except Exception as e:
        st.error(f"An unexpected error occurred during update: {e}")

# --- Streamlit App UI (MODIFIED for Confidence Score Display) ---

def main():
    st.set_page_config(page_title="NRC Document Data Extractor", layout="centered")

    st.title("🇲🇲 NRC Document Data Extractor (HITL & Validation)")
    st.markdown("This tool uses an AI to extract data from a Myanmar NRC image and provides a human-in-the-loop step for correction.")

    # Initialize session state for data storage
    if 'extracted_data' not in st.session_state:
        st.session_state['extracted_data'] = None
    if 'json_text' not in st.session_state:
        st.session_state['json_text'] = None

    # --- INPUT SELECTION: Use tabs for file upload or camera input ---
    tab1, tab2 = st.tabs(["🖼️ Upload Image", "📸 Take Photo"])

    uploaded_file = None
    camera_image = None
    selected_source = None

    with tab1:
        uploaded_file = st.file_uploader(
            "Upload NRC Image (PNG, JPG)",
            type=["png", "jpg", "jpeg"]
        )
    
    with tab2:
        camera_image = st.camera_input("Snap a photo of the NRC Document")
    
    # Determine the source of the image
    selected_source = uploaded_file if uploaded_file is not None else camera_image

    if selected_source is not None:
        image_bytes = selected_source.getvalue()
        st.image(image_bytes, caption='Input NRC Document', use_column_width=True)
        st.divider()

        # Button to trigger extraction
        if st.button("1. Extract & Validate Data", type="primary"):
            # Clear previous data to ensure fresh extraction
            st.session_state['extracted_data'] = None
            st.session_state['json_text'] = None

            extracted_data = extract_nrc_data(image_bytes)
            if extracted_data:
                st.session_state['extracted_data'] = extracted_data
                # Store the JSON string for the editor
                st.session_state['json_text'] = json.dumps(extracted_data, indent=2)
                st.rerun() 
                
        # --- NEW STEPS: Display Results, Validation, and Correction ---
        if st.session_state['extracted_data'] is not None:
            data = st.session_state['extracted_data']
            warnings = data.get('validation_warnings', [])
            confidence = data.get('Overall_Confidence_Score')
            
            # --- Confidence Score Display ---
            if confidence is not None:
                confidence_percent = f"{confidence * 100:.0f}%"
                
                if confidence > 0.8:
                    st.success(f"**Confidence Score:** {confidence_percent} (High Certainty) ✅")
                elif confidence > 0.5:
                    st.warning(f"**Confidence Score:** {confidence_percent} (Moderate Certainty) ⚠️")
                else:
                    st.error(f"**Confidence Score:** {confidence_percent} (Low Certainty) ❌ - **Human review is highly recommended.**")
            
            st.markdown("---")
            
            # 2. Display Warnings
            st.subheader("⚠️ Validation Warnings")
            if warnings:
                st.warning("The extracted data has the following potential errors:")
                for warning in warnings:
                    st.markdown(f"- **{warning}**")
            else:
                st.success("Data passed all preliminary validation checks.")
            
            st.markdown("---")

            # 3. Human-in-the-Loop Correction (Editable JSON)
            st.subheader("2. Human-in-the-Loop Correction (Editable JSON)")
            st.markdown("Review the raw output below. If the model misread any Burmese handwriting, correct the values in the JSON. **Then click 'Update'**.")
            
            # Prepare data for the editor (removing warnings for a cleaner input)
            editable_data = data.copy()
            editable_data.pop('validation_warnings', None)
            
            edited_json_text = st.text_area(
                "Edit Raw JSON Output",
                st.session_state.get('json_text', json.dumps(editable_data, indent=2)),
                height=350,
                key='json_input'
            )
            
            # Button to parse and validate the user's changes
            if st.button("Update and Re-Validate Extracted Data", type="secondary"):
                update_data_from_json(edited_json_text)
                st.rerun()

            # 4. Final Data Display and Download
            st.subheader("3. Final Corrected Data")

            # Display results in a clear table format
            final_data_view = [
                ("Confidence Score", f"{data.get('Overall_Confidence_Score', 0.0) * 100:.0f}%"), # Display percentage
                ("NRC State Code", data.get("NRC_State_Code", "N/A")),
                ("NRC Number (အမှတ်)", data.get("NRC_Number", "N/A")),
                ("Name (အမည်)", data.get("Name", "N/A")),
                ("Father's Name (အဘအမည်)", data.get("Fathers_Name", "N/A")),
                ("Date of Birth (မွေးသက္ကရာဇ်)", data.get("Date_of_Birth", "N/A")),
                ("Height (အရပ်)", data.get("Height", "N/A")),
                ("Religion (ကိုးကွယ်သည့်ဘာသာ)", data.get("Religion", "N/A")),
                ("Blood Type (သွေးအမျိုးအစား)", data.get("Blood_Type", "N/A"))
            ]
            st.table(final_data_view)
            
            # Download Button
            # Remove the 'validation_warnings' field for the final output file
            download_data = data.copy()
            download_data.pop('validation_warnings', None)
            
            json_output = json.dumps(download_data, indent=2)
            
            # Prepare CSV output
            header = list(download_data.keys())
            values = [str(download_data[k]) for k in header]
            csv_output = ",".join(header) + "\n" + ",".join(values)

            st.download_button(
                label="⬇️ Download Final JSON",
                data=json_output,
                file_name="nrc_data_corrected.json",
                mime="application/json",
            )
            st.download_button(
                label="⬇️ Download Final CSV",
                data=csv_output,
                file_name="nrc_data_corrected.csv",
                mime="text/csv",
            )


    else:
        st.info("Please upload an image file or take a photo and click '1. Extract & Validate Data' to begin.")

if __name__ == "__main__":
    main()