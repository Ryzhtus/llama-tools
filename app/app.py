import streamlit as st
import requests
import pypdfium2 as pdfium
from src.functions import functions_map, parse_function_call

llm_url = "http://llm_app:8001"
retrieval_url = "http://retrieval_app:8002"


def clear_application_data():
    # Clear chat history
    if "chat_history" in st.session_state:
        del st.session_state.chat_history

    # Clear all uploaded documents
    if "uploaded_docs" in st.session_state:
        del st.session_state.uploaded_docs

    # Remove the selected document
    if "selected_doc" in st.session_state:
        del st.session_state.selected_doc

    requests.post(url=llm_url + "/reset")
    requests.post(url=retrieval_url + "/reset")

    # Optional: add a message to confirm clearing is done
    st.success("All data cleared successfully!")


def upload_file(file):
    # Since we're dealing with an UploadedFile object, we need to reset its position
    file.seek(0)  # Reset file pointer to the beginning
    files = {
        "file": (
            file.name,
            file.getvalue(),  # Get the file buffer
            "multipart/form-data",
        )
    }
    response = requests.post(url=retrieval_url + "/add_document", files=files)
    return response


def function_call(function_call_str: str) -> str:
    parsed_call = parse_function_call(function_call_str)
    func_name = parsed_call.get("name")
    func_args = parsed_call.get("arguments")

    if func_name == "get_document_text":
        function_response = functions_map[func_name](
            st.session_state["selected_document_name"]
        )
    else:
        function_response = functions_map[func_name](**func_args)

    return str(function_response)


def generate_response(prompt: str):
    # Simulate a POST request to a backend that processes the user input
    llm_response = requests.post(llm_url + "/generate", json={"prompt": prompt})
    llm_response_str = llm_response.json()["response"]

    llm_input_str = function_call(llm_response_str)
    if llm_input_str != None:
        llm_response = requests.post(
            llm_url + "/generate", json={"prompt": llm_input_str}
        )
        llm_response_str = llm_response.json()["response"]

    return llm_response_str


supplementary_container = st.sidebar

with supplementary_container:
    if st.button("Reset Session"):
        clear_application_data()

    st.title("Documents Collection")

    if "selected_document_name" in st.session_state:
        st.write(f"Selected: {st.session_state['selected_document_name']}")

    uploaded_files = st.file_uploader("Load a document", accept_multiple_files=True)

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

                    else:
                        st.write("Preview not available for this file type.")
                except Exception as e:
                    st.write("Error loading preview:", e)

# Create the main container for text input/output
chat_container = st.container()

with chat_container:
    st.title("Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.markdown(generate_response(prompt))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
