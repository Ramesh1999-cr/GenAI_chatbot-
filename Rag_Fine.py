import streamlit as st
import chromadb
from langchain.llms import Ollama

# Sample dataset
faq_data = [
    {"question": "What is net income?", "answer": "Net income is the total profit of a company after expenses."},
    {"question": "What is revenue?",
     "answer": "Revenue is the total income generated from normal business operations."},
    {"question": "Define liabilities.", "answer": "Liabilities are what a company owes to external parties."}
]


# Set up ChromaDB for Retrieval-Augmented Generation (RAG)
def setup_chromadb():
    client = chromadb.Client()
    collection = client.get_or_create_collection("financial_faq")

    # Adding documents to the collection
    for item in faq_data:
        collection.add(ids=[item["question"]], documents=[item["answer"]])
    return collection


# RAG function
def get_rag_response(query, collection):
    # Retrieve documents based on similarity to query
    results = collection.query(query_texts=[query], n_results=1)
    context = " ".join(results['documents'][0]) if results['documents'] else "No relevant data found."

    # Using Ollama model for response generation with context
    llm = Ollama(model="gemma:2b")
    response = llm(f"Context: {context}\nQuery: {query}")
    return response


# Fine-tuning simulation (for illustration, actual fine-tuning requires specialized setup)
def get_finetuned_response(query):
    llm = Ollama(model="gemma:2b")  # Placeholder for fine-tuned model
    response = llm(query)
    return response


# Streamlit Interface
st.title("RAG vs. Fine-tuning Comparison")

# Input for query
query = st.text_input("Enter your query here:")

# Run comparison when query is entered
if query:
    # Setup ChromaDB collection
    collection = setup_chromadb()

    # Get responses
    rag_response = get_rag_response(query, collection)
    finetune_response = get_finetuned_response(query)

    # Display results
    st.subheader("Results")
    st.write("**RAG Response:**", rag_response)
    st.write("**Fine-tuned Model Response:**", finetune_response)
