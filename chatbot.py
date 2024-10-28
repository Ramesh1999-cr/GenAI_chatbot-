import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

#from openai.error import RateLimitError
import time

# Your OpenAI API key
OPENAI_API_KEY = "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"

# Streamlit App
st.header("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF, CSV, or Excel file and start asking questions", type=["pdf", "csv", "xlsx"])

# Extract the text or data
if file is not None:
    # Determine the file type
    if file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Check if text is extracted
        if not text.strip():
            st.error("The PDF appears to be empty or not text-based.")
        else:
            # Break it into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n"],
                chunk_size=1000,
                chunk_overlap=150
            )
            chunks = text_splitter.split_text(text)

    elif file.type == "text/csv":
        df = pd.read_csv(file)
        text = df.to_string(index=False)

        # Break it into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n"],
            chunk_size=1000,
            chunk_overlap=150
        )
        chunks = text_splitter.split_text(text)

    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(file)
        text = df.to_string(index=False)

        # Break it into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n"],
            chunk_size=1000,
            chunk_overlap=150
        )
        chunks = text_splitter.split_text(text)

    # Log the chunks for debugging
    st.write("Text Chunks:", chunks)

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)




    # def handle_rate_limit():
    #     retries = 3
    #     wait_time = 60
    #     for attempt in range(retries):
    #         try:
    #             # Your API call here
    #             pass
    #         except RateLimitError:
    #             if attempt < retries - 1:
    #                 print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
    #                 time.sleep(wait_time)
    #                 wait_time *= 2  # Exponential backoff
    #             else:
    #                 print("Rate limit exceeded and retries exhausted.")
    #                 raise
    #

    # Creating vector store - FAISS using text chunks and the embeddings object
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get user question
    user_question = st.text_input("Type Your question here")

    # Do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)

        # Define the LLM
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )

        # Output results
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)
        st.write(response)


























# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.chat_models import ChatOpenAI
#
# # Your OpenAI API key
# OPENAI_API_KEY = "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"
#
# # Streamlit App
# st.header("My first Chatbot")
#
# with st.sidebar:
#     st.title("Your Documents")
#     file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")
#
# # Extract the text
# if file is not None:
#     pdf_reader = PdfReader(file)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#
#     # Check if text is extracted
#     if not text.strip():
#         st.error("The PDF appears to be empty or not text-based.")
#     else:
#         # Break it into chunks
#         text_splitter = RecursiveCharacterTextSplitter(
#             separators=["\n"],
#             chunk_size=1000,
#             chunk_overlap=150
#         )
#         chunks = text_splitter.split_text(text)
#
#         # Log the chunks for debugging
#         st.write("Text Chunks:", chunks)
#
#         # Initialize OpenAI embeddings
#         embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#
#         # Creating vector store - FAISS using text chunks and the embeddings object
#         vector_store = FAISS.from_texts(chunks, embeddings)
#
#         # Get user question
#         user_question = st.text_input("Type Your question here")
#
#         # Do similarity search
#         if user_question:
#             match = vector_store.similarity_search(user_question)
#
#             # Define the LLM
#             llm = ChatOpenAI(
#                 openai_api_key=OPENAI_API_KEY,
#                 temperature=0,
#                 max_tokens=1000,
#                 model_name="gpt-3.5-turbo"
#             )
#
#             # Output results
#             chain = load_qa_chain(llm, chain_type="stuff")
#             response = chain.run(input_documents=match, question=user_question)
#             st.write(response)
#
#
#
#
