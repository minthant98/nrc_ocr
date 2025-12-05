import streamlit as st
import os
import io
import json
import base64
import difflib
from datetime import date
from PIL import Image, ExifTags
from typing import Dict, Any, List, Optional

# --- IMPORT CORE LOGIC ---
from nrc_ocr_processor import process_image, extract_nrc_data_augmented, image_to_bytes
# Note: You may need to create a simple nrc_ocr_processor.py file with just dummy functions
# if you are running this locally without the actual model setup yet.

# --- API KEY SETUP (Keep this in the main app) ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    # Fallback: If the key isn't in st.secrets, check environment variables.
    API_KEY = os.environ.get("GEMINI_API_KEY", "")

# --- Utility Functions (Kept in app.py as they rely on st.session_state, date, and constants) ---

def is_valid_date(burmese_date_str: str) -> bool:
    """Tries to parse a date from the Burmese string and validates it."""
    today = date.today()
    import re
    # Look for a 4-digit number (representing the year)
    year_match = re.search(r'(\d{4})', burmese_date_str)
    if year_match:
        try:
            year = int(year_match.group(1))
            # Basic plausibility checks
            if year > today.year: return False
            if today.year - year > 150: return False # Max age 150 years
            return True
        except ValueError:
            return False
    # If no Latin year found, we assume it's entirely in Burmese script and cannot validate the year easily
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
    # Check if it looks like (A), (C), or (N) including parentheses
    if nrc_sth and not (nrc_sth.startswith('(') and nrc_sth.endswith(')') and len(nrc_sth) >= 3):
        warnings.append(f"NRC Classification Code ('NRC_sth') '{nrc_sth}' should be formatted as (C), (N), or (A) and include parentheses.")

    # 4. Validate Date of Birth
    dob_burmese = data.get('Date_of_Birth', '').strip()
    if dob_burmese and not is_valid_date(dob_burmese):
        warnings.append(f"Date of Birth '{dob_burmese}' appears suspicious (e.g., future date or over 150 years old, based on Latin digits if found).")
        
    return warnings

def calculate_accuracy_score(original_data: Dict[str, Any], corrected_data: Dict[str, Any]) -> float:
    """Calculates field-level similarity between model's output and human-corrected output."""
    fields_to_compare = [
        "NRC_state_division", "NRC_township", "NRC_sth", "NRC_no", 
        "Name", "Fathers_Name", "Date_of_Birth"
    ]
    total_score = 0.0
    
    for field in fields_to_compare:
        original_value = str(original_data.get(field, "")).strip()
        corrected_value = str(corrected_data.get(field, "")).strip()
        
        if original_value or corrected_value:
            # difflib.SequenceMatcher is a good measure for string similarity
            similarity_ratio = difflib.SequenceMatcher(None, original_value, corrected_value).ratio()
            total_score += similarity_ratio
        else:
            total_score += 1.0 # Perfect match if both are empty
            
    accuracy = total_score / len(fields_to_compare)
    return accuracy

def update_data_from_fields(fields_data: Dict[str, str]):
    """
    Updates the session state with human-corrected data from the form, re-validates, 
    and recalculates the accuracy score.
    """
    updated_data = st.session_state['extracted_data'].copy() 
    for key, value in fields_data.items():
        updated_data[key] = value

    st.session_state['extracted_data'] = updated_data
    
    # Re-validate the corrected data
    updated_data['validation_warnings'] = validate_nrc_data(updated_data)

    # Recalculate accuracy against the original model output
    original_data = st.session_state['original_data']
    accuracy = calculate_accuracy_score(original_data, updated_data)
    st.session_state['accuracy_score'] = accuracy
    
    st.success("Data successfully updated, re-validated, and accuracy score recalculated.")
    st.rerun() # Rerun to update the metrics and display

def rotate_uploaded_image(angle: int):
    """
    Applies manual rotation to the original uploaded image bytes and stores them.
    This modifies the image source before enhancement.
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
        
        # Save the new bytes back to session state for the next run
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
    st.set_page_config(page_title="NRC Document Data Extractor V7 R (Augmented)", layout="centered")

    st.title("🇲🇲 NRC Document Data Extractor V7 R (Augmented)")
    st.subheader("Combined Local CNN (91% Accurate) and Gemini Vision for OCR")
    st.markdown("This version uses a highly accurate local model to pre-extract handwritten Burmese fields, which are then passed to the Gemini API as high-confidence hints.")

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
            # This is the original, un-rotated, un-enhanced source image bytes
            st.session_state['uploaded_file_bytes'] = uploaded_file.getvalue() 
            st.session_state['uploaded_file_name'] = file_name_or_default
            st.session_state['current_image_bytes'] = uploaded_file.getvalue() 

    
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
            st.image(image_bytes_to_display, caption='Current Input Document', use_column_width=True)

        with img_tab2:
            if 'enhanced_image_bytes' in st.session_state and st.session_state['enhanced_image_bytes'] is not None:
                st.image(st.session_state['enhanced_image_bytes'], caption='Sharpened (2.0x) and Contrasted (1.5x)', use_column_width=True)
            else:
                st.info("The AI processed image preview will appear here after extraction.")

        st.divider()

        # Button to trigger extraction
        if st.button("1. Extract & Validate Data", type="primary"):
            
            # 1. Clear previous results
            st.session_state['extracted_data'] = None 
            st.session_state['accuracy_score'] = None
            
            with st.spinner("Processing image (Enhancement, Local CNN, and Gemini API call)..."):
                
                # 2. Run the image through the pipeline for enhancement and orientation
                # The original bytes include the last manual rotation applied
                enhanced_bytes = process_image(st.session_state['uploaded_file_bytes']) 
                st.session_state['enhanced_image_bytes'] = enhanced_bytes
                
                # 3. Perform the Augmented Extraction (calling the imported function)
                # We pass the original bytes to the processor for the local CNN to use
                extracted_data = extract_nrc_data_augmented(
                    enhanced_bytes, 
                    api_key=API_KEY,
                    original_image_bytes=st.session_state['uploaded_file_bytes']
                )
            
            if extracted_data:
                st.session_state['original_data'] = extracted_data.copy()
                warnings = validate_nrc_data(extracted_data)
                extracted_data['validation_warnings'] = warnings
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

            # 3. Final Data Display and Download
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
                    file_name="nrc_data_corrected_augmented.json",
                    mime="application/json",
                )
            with col_dl2:
                st.download_button(
                    label="⬇️ Download Final CSV",
                    data=csv_output,
                    file_name="nrc_data_corrected_augmented.csv",
                    mime="text/csv",
                )

            st.markdown("---")

            # 4. Human-in-the-Loop Correction (Text Fields)
            st.subheader("✍️ 4. Correct Data & Recalculate Accuracy")
            st.markdown("Review and correct any errors below. Any change will be used to calculate the model's accuracy.")
            
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
                    # update_data_from_fields triggers st.rerun()
                    update_data_from_fields(fields_data) 
            
            st.markdown("---")

    else:
        st.info("Please upload an image file or take a photo and click '1. Extract & Validate Data' to begin. Use the rotation tools if your image is not upright.")

if __name__ == "__main__":
    main()