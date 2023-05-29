import streamlit as st
import os
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import textract
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone 

# OPENAI API KEY
os.environ["OPENAI_API_KEY"] = "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"

# Pinecone 
PINECONE_API_KEY = '78fa8724-5c6a-4105-859b-4aac8d590171'
PINECONE_ENV = 'asia-southeast1-gcp'

# function to retrieve embeddings
def retrieve_embeddings(file_path, fileName):
    # converting pdf to text
    doc = textract.process(file_path)

    # Create the directory if it doesn't exist
    if not os.path.exists("data_pdfTotxtFiles"):
        os.makedirs("data_pdfTotxtFiles")

    # Check if the file already exists
    file_path = os.path.join("data_pdfTotxtFiles", fileName + ".txt")

    # saving to .txt
    with open(file_path, 'w', encoding="utf-8") as f:
        f.write(doc.decode('utf-8'))

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # counting the tokens
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))
    
    # splitting text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=24, length_function=count_tokens)

    chunks = text_splitter.create_documents([text])

    # getting embeddings
    embeddings = OpenAIEmbeddings()

    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV  # next to api key in console
    )

    index_name = "new"

    docsearch = Pinecone.from_documents(chunks, embeddings, index_name=index_name)

    # if you already have an index, you can load it like this
    # docsearch = Pinecone.from_existing_index(index_name, embeddings)

    return docsearch

def main ():

    # print('hello')

    st.title("Upload pdf files to generate embeddings!")

    # Create the directory if it doesn't exist
    if not os.path.exists("data_pdfFiles"):
        os.makedirs("data_pdfFiles")

    # Create a file uploader widget
    # returns a list og files uploaded with (id, name, type, size)
    uploaded_files = st.file_uploader("Upload a file", accept_multiple_files=True)

    # Check if a file was uploaded
    if uploaded_files is not None:
        
        st.write(uploaded_files)
        # Display file details
        upload_mssg = st.success("Files Uploaded Successfully!")
        time.sleep(3)
        upload_mssg.empty()

        # Process each uploaded file
        for uploaded_file in uploaded_files:
            # Check if the file already exists
            file_path = os.path.join("data_pdfFiles", uploaded_file.name)
            if os.path.exists(file_path):
                st.warning("File already exists: {}".format(uploaded_file.name))
                
            else:
                # Save the file to the data_pdfFiles directory
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                savedFile_mssg = st.success("Saved File: {}".format(uploaded_file.name))
                time.sleep(3)
                savedFile_mssg.empty()

                db = retrieve_embeddings(file_path, uploaded_file.name)

                print(db)

                indexGen_mssg = st.success("Index stored in Supabase for" + uploaded_file.name + "!")
                time.sleep(3)
                indexGen_mssg.empty()

if __name__ == "__main__":
    main()