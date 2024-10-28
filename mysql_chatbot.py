import streamlit as st
import mysql.connector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# Your OpenAI API key
OPENAI_API_KEY = "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"

# MySQL Database connection
def connect_to_db():
    return mysql.connector.connect(
        host="localhost",  # Replace with your MySQL host
        port=3306,
        user="root",  # Replace with your MySQL username
        password="root",  # Replace with your MySQL password
        database="bank"  # Replace with your MySQL database name
    )

def get_text_data(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM bank.bank_app_bankbalance")  # Update column name if necessary
    return [row[0] for row in cursor.fetchall()]

# Streamlit App
st.header("My First Chatbot")

# Connect to the MySQL database and retrieve text data
conn = connect_to_db()
text_data = get_text_data(conn)
conn.close()

if not text_data:
    st.error("No data found in the database.")
else:
    # Combine all text data into a single string
    combined_text = "\n".join(text_data)

    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_text(combined_text)

    # Log the chunks for debugging
    st.write("Text Chunks:", chunks)

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

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
