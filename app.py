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
    mime_type = "image/png"  # Assuming the image is a PNG

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
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        # 4. Process the response
        result = response.json()
        
        # Extract the JSON string from the response
        # Navigate the potentially complex response structure
        json_string = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
        
        if json_string:
            # Parse the JSON string into a Python dictionary
            return json.loads(json_string)
        else:
            # Try to get detailed error info if no content was generated
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


# --- Streamlit App UI ---
def main():
    st.set_page_config(page_title="NRC Document Data Extractor", layout="centered")

    st.title("MM NRC Document Data Extractor")
    st.markdown("Upload a photo or scan of a Myanmar National Registration Card (NRC) to extract key details using the **Gemini AI model** with structured JSON output. **Tip:** For best results with handwritten Burmese, ensure the uploaded image is high-resolution and clearly focused.")

    # File Uploader component
    uploaded_file = st.file_uploader(
        "Upload NRC Image (PNG, JPG)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image_bytes = uploaded_file.getvalue()
        st.image(image_bytes, caption='Uploaded NRC Document', use_column_width=True)
        st.divider()

        # Button to trigger extraction
        if st.button("Extract Data", type="primary"):
            # Clear previous data to ensure fresh extraction
            if 'extracted_data' in st.session_state:
                del st.session_state['extracted_data']

            with st.spinner("Analyzing document and extracting data..."):
                extracted_data = extract_nrc_data(image_bytes)
                st.session_state['extracted_data'] = extracted_data

        # Display the results if extraction was successful
        if 'extracted_data' in st.session_state and st.session_state['extracted_data'] is not None:
            data = st.session_state['extracted_data']
            st.success("✅ Data Extraction Complete!")
            
            # Display results in a clear table format
            st.subheader("Extracted Information")
            
            # Convert the dictionary to a list of tuples for table display
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
            
            st.subheader("Raw JSON Output")
            st.json(data)
            
    else:
        st.info("Please upload an image file to begin data extraction.")

if __name__ == "__main__":
    main()