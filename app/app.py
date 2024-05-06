import streamlit as st
import requests
import pypdfium2 as pdfium
import json
from src.functions import functions_map, check_function_call, parse_function_call

llm_url = "http://llm_app:8001"
retrieval_url = "http://retrieval_app:8002"


def function_call(function_call_str: str) -> str:
    func_name, func_args = parse_function_call(function_call_str)
    func_args = json.loads(func_args)

    if func_name == "get_document_text":
        function_response = functions_map[func_name](
            st.session_state["selected_document_name"]
        )
    else:
        function_response = functions_map[func_name](**func_args)

    return function_response


def upload_file(file):
    # Since we're dealing with an UploadedFile object, we need to reset its position
    # file.seek(0)  # Reset file pointer to the beginning
    files = {
        "file": (
            file.name,
            file.getvalue(),  # Get the file buffer
            "multipart/form-data",
        )
    }
    response = requests.post(url=retrieval_url + "/add_document", files=files)
    return response


supplementary_container = st.sidebar

with supplementary_container:
    st.title("Documents Collection")
    uploaded_files = st.file_uploader(
        "Documents Collection", accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_container = st.container()
            with file_container:
                # Display the filename and a clickable link or button for selection
                file_info = f"Filename: {uploaded_file.name}"
                st.write(file_info)

                # Attempt to display a file preview based on the file type
                try:
                    if uploaded_file.type == "application/pdf":
                        pdf = pdfium.PdfDocument(uploaded_file)
                        upload_file(uploaded_file)

                        # Use the first page for preview
                        page = pdf[0]  # 0 is the first page
                        image = page.render(scale=4).to_pil()

                        # Display the image
                        st.image(
                            image, caption=uploaded_file.name, use_column_width=True
                        )

                        # Create a button for selecting the document
                        if st.button(f"Select {uploaded_file.name}"):
                            st.session_state["selected_document_name"] = (
                                uploaded_file.name
                            )
                            st.write(f"Selected: {uploaded_file.name}")
                    else:
                        st.write("Preview not available for this file type.")
                except Exception as e:
                    st.write("Error loading preview:", e)

# Create the main container for text input/output
main_container = st.container()
with main_container:
    st.title("Main Container")
    user_input = st.text_area("Enter your text here:")

    if user_input:
        # Simulate a POST request to a backend that processes the user input
        llm_response = requests.post(llm_url + "/generate", json={"prompt": user_input})
        llm_response_str = llm_response.json()["response"]

        if check_function_call(llm_response_str):
            llm_input_str = function_call(llm_response_str)
            llm_response = requests.post(
                llm_url + "/generate", json={"prompt": llm_input_str}
            )
            llm_response_str = llm_response.json()["response"]
        # Display response from backend assuming it returns JSON
        st.write(llm_response_str)

# Optionally display the selected document name outside the loop
if "selected_document_name" in st.session_state:
    st.write(f"Selected document: {st.session_state['selected_document_name']}")
