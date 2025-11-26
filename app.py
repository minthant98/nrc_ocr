import streamlit as st
import requests
import base64
import json
import io
import os # Required for checking environment variables

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

# Define the mandatory structured output schema for clear data extraction
JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
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
    # --- CHANGE: Only core, easily found fields are now REQUIRED ---
    "required": ["NRC_Number", "Name", "Fathers_Name", "Date_of_Birth"] 
}

# Define the system instruction to guide the model's behavior and role
SYSTEM_INSTRUCTION = (
    "You are an expert Optical Character Recognition (OCR) and Intelligent Document "
    "Processing (IDP) system specialized in reading Myanmar National Registration Card (NRC) documents. "
    "Your primary task is to accurately extract the required fields, prioritizing the precise recognition "
    "of *handwritten* Burmese script, even when the image quality is imperfect or the script is ambiguous. "
    "Output the results ONLY as a JSON object conforming to the provided schema."
)

# --- Function to call the Gemini API ---
@st.cache_data(show_spinner=False)
def extract_nrc_data(image_bytes: bytes):
    """
    Calls the Gemini API to extract structured data from the image bytes.
    """
    if not API_KEY:
        st.error("API Key is missing. Please configure 'GEMINI_API_KEY' in your environment.")
        return None
        
    st.info("Sending document to Gemini API for extraction. This may take a moment...")
    
    # 1. Prepare the image data (base64 encoding)
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    mime_type = "image/png"

    # 2. Construct the API Payload
    user_query = (
        "Analyze the provided Myanmar NRC document image. Locate and extract the following fields "
        "and their values: 1) The full NRC number (အမှတ်), 2) The cardholder's Name (အမည်), "
        "3) The Father's Name (အဘအမည်), 4) The Date of Birth (မွေးသက္ကရာဇ်), 5) Height (အရပ်), "
        "6) Religion (ကိုးကွယ်သည့်ဘာသာ), and 7) Blood Type (သွေးအမျိုးအစား). "
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
            return json.loads(json_string)
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

# --- Custom Logic to Handle JSON Updates ---

def update_data_from_json(json_text):
    """Parses the edited JSON text and updates the session state."""
    try:
        updated_data = json.loads(json_text)
        
        # Simple validation: Check if it's a dict and has required fields
        if not isinstance(updated_data, dict) or not all(field in updated_data for field in JSON_SCHEMA['required']):
            st.error("Error: The updated JSON must be an object containing at least NRC_Number, Name, Fathers_Name, and Date_of_Birth.")
            return

        st.session_state['extracted_data'] = updated_data
        st.session_state['json_text'] = json.dumps(updated_data, indent=2)
        st.success("✅ Data successfully updated from JSON editor.")
    except json.JSONDecodeError:
        st.error("Error: Invalid JSON format. Please check syntax (e.g., quotes, commas, brackets).")
    except Exception as e:
        st.error(f"An unexpected error occurred during update: {e}")

# --- Streamlit App UI ---
def main():
    st.set_page_config(page_title="NRC Document Data Extractor", layout="centered")

    st.title("🇲🇲 NRC Document Data Extractor")
    st.markdown("Use the options below to either upload an existing image or take a new photo. **Tip:** For best results with handwritten Burmese, ensure the image is high-resolution and clearly focused.")

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
        if st.button("1. Extract Data from Image", type="primary"):
            # Clear previous data to ensure fresh extraction
            extracted_data = extract_nrc_data(image_bytes)
            if extracted_data:
                st.session_state['extracted_data'] = extracted_data
                # Store the JSON string for the editor
                st.session_state['json_text'] = json.dumps(extracted_data, indent=2)

        # Display the results and editor if data exists
        if 'extracted_data' in st.session_state and st.session_state['extracted_data'] is not None:
            data = st.session_state['extracted_data']
            
            st.subheader("Extracted Information Table (View Only)")
            
            # Display results in a clear table format
            display_data = [
                ("NRC Number (အမှတ်)", data.get("NRC_Number", "N/A")),
                ("Name (အမည်)", data.get("Name", "N/A")),
                ("Father's Name (အဘအမည်)", data.get("Fathers_Name", "N/A")),
                ("Date of Birth (မွေးသက္ကရာဇ်)", data.get("Date_of_Birth", "N/A")),
                ("Height (အရပ်)", data.get("Height", "N/A")),
                ("Religion (ကိုးကွယ်သည့်ဘာသာ)", data.get("Religion", "N/A")),
                ("Blood Type (သွေးအမျိုးအစား)", data.get("Blood_Type", "N/A"))
            ]
            
            st.table(display_data)

            st.subheader("2. Human-in-the-Loop Correction (Editable JSON)")
            st.markdown("If the model made errors, correct the values in the text area below.")
            
            # Use st.text_area to allow editing of the raw JSON string
            edited_json_text = st.text_area(
                "Edit Raw JSON Output",
                st.session_state.get('json_text', json.dumps(data, indent=2)),
                height=350,
                key='json_input'
            )
            
            # Button to parse and validate the user's changes
            if st.button("Update Extracted Data", type="secondary"):
                update_data_from_json(edited_json_text)
                # Re-run the script to reflect the changes in the table immediately
                st.rerun()

    else:
        st.info("Please upload an image file or take a photo and click '1. Extract Data from Image' to begin.")

if __name__ == "__main__":
    main()