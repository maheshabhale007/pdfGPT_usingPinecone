import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import pinecone
import os
from langchain.vectorstores import Pinecone

import os

os.environ["OPENAI_API_KEY"] = "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"

PINECONE_API_KEY = '78fa8724-5c6a-4105-859b-4aac8d590171'
PINECONE_ENV = 'asia-southeast1-gcp'

def main():

    # print('hello')

    st.title("SprihGPT")

    embeddings = OpenAIEmbeddings()

    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV  # next to api key in console
    )

    index_name = "new"

    # if you already have an index, you can load it like this
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    # prompt question
    query = st.text_input("Ask Question:")

    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

    docs = docsearch.similarity_search(query)

    response = chain.run(input_documents=docs, question=query)

    print(response)
    print('\n')

    st.write(response)

if __name__ == '__main__':
    main()