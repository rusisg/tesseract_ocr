import streamlit as st
from PIL import Image
import json
import pandas as pd

# --- Import custom modules ---
from file_handler import convert_pdf_to_images
from donut_parser import load_donut_model, parse_document
from paddle_parser import load_paddle_model, extract_full_text
from post_processor import preprocess_image_for_ocr, load_llm_pipeline, fuse_and_clean_with_llm


# --- Model Loading ---
@st.cache_resource
def load_all_models():
    donut_processor, donut_model, device = load_donut_model()
    paddle_model = load_paddle_model()
    llm_pipeline = load_llm_pipeline()
    return donut_processor, donut_model, device, paddle_model, llm_pipeline


st.set_page_config(layout="wide", page_title="Multi-Format Document Extractor")
donut_processor, donut_model, device, paddle_model, llm_pipeline = load_all_models()

# --- Streamlit App UI ---
st.title("📄 Multi-Format Document Extractor (Image, PDF)")
st.markdown("Upload a document. The system will auto-detect the type and extract structured data using the 3-stage AI pipeline.")

# Updated file uploader to accept more types
uploaded_file = st.file_uploader(
    "Choose a document (Image or PDF)", 
    type=["png", "jpg", "jpeg", "pdf", "csv", "xlsx"]
)

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension in ["png", "jpg", "jpeg"]:
        st.info("Image file detected. Processing...")
        images_to_process = [Image.open(uploaded_file).convert("RGB")]
    
    elif file_extension == "pdf":
        st.info("PDF file detected. Converting pages to images...")
        file_bytes = uploaded_file.getvalue()
        images_to_process = convert_pdf_to_images(file_bytes)
        st.success(f"PDF converted successfully. Found {len(images_to_process)} pages.")
    
    elif file_extension in ["csv", "xlsx"]:
        st.info("Excel/CSV schema file detected. Displaying contents.")
        try:
            df = pd.read_csv(uploaded_file) if file_extension == "csv" else pd.read_excel(uploaded_file)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
        images_to_process = []

    else:
        st.error("Unsupported file type.")
        images_to_process = []

    if images_to_process:
        page_selection = 0
        if len(images_to_process) > 1:
            page_selection = st.selectbox(
                f"This document has {len(images_to_process)} pages. Select a page to view and process:", 
                options=range(len(images_to_process)), 
                format_func=lambda x: f"Page {x + 1}"
            )

        original_image = images_to_process[page_selection]
        
        st.header(f"Stage 1: Pre-processing (Page {page_selection + 1})")
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Original Image", use_column_width=True)
        processed_image = preprocess_image_for_ocr(original_image)
        with col2:
            st.image(processed_image, caption="Processed Image (for Donut Model)", use_column_width=True)

        # --- Stage 2 & 3 ---
        if st.button(f"Extract & Fuse Data for Page {page_selection + 1}", type="primary"):
            st.header("Stage 2: Parallel Extraction")
            col1, col2 = st.columns(2)
            
            with st.spinner("Running Donut & PaddleOCR models..."):
                donut_output = parse_document(processed_image, donut_processor, donut_model, device)
                paddle_output = extract_full_text(original_image, paddle_model)

            with col1:
                st.subheader("Donut Model Output (Structured Guess)")
                st.json(donut_output)
            
            with col2:
                st.subheader("PaddleOCR Output (Raw Text)")
                st.text("\n".join(paddle_output))

            st.header("Stage 3: LLM Fusion & Correction")
            with st.spinner("Fusing results with local LLM..."):
                final_output = fuse_and_clean_with_llm(llm_pipeline, donut_output, paddle_output)
            
            st.success("Fusion complete! Here is the final, high-accuracy JSON.")
            st.json(final_output)

            st.download_button(
                label="Download Final JSON",
                data=json.dumps(final_output, indent=4),
                file_name=f"page_{page_selection + 1}_data.json",
                mime="application/json",
            )
else:
    st.info("Upload a document to see the pipeline in action.")