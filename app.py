import streamlit as st 
import os
import time
import requests
import numpy as np
import faiss
import pickle
from bs4 import BeautifulSoup
from mistralai import Mistral
from mistralai.models import UserMessage

# Set page config at the very beginning
st.set_page_config(page_title="UDST Policy Chatbot", layout="wide")

# Load API Key from Streamlit Secrets
try:
    API_KEY = st.secrets["MISTRAL_API_KEY"]
    if not API_KEY:
        raise ValueError("API Key not found in Streamlit secrets.")
except Exception as e:
    st.error(f"Error loading API Key: {e}")
    st.stop()

client = Mistral(api_key=API_KEY)

# Define paths for FAISS index and chunks
FAISS_INDEX_PATH = "policy_embeddings.index"
ALL_CHUNKS_PATH = "all_chunks.pkl"

# Function to fetch policy data
def fetch_policies():
    return {
        "Graduation Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",
        "Graduate Admissions Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-admissions-policy",
        "Graduate Academic Standing Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-policy",
        "Credit Hour Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
        "Graduate Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-policy",
        "Graduate Final Grade Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-procedure",
        "International Student Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-policy",
        "International Student Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-procedure",
        "Registration Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy",
        "Registration Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-procedure",
        # Newly Added Policies
        "Academic Annual Leave Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-annual-leave-policy",
        "Academic Appraisal Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-policy",
        "Academic Credentials Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-credentials-policy",
        "Academic Freedom Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-freedom-policy",
        "Academic Members’ Retention Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-members’-retention-policy",
        "Academic Professional Development Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-professional-development",
        "Academic Qualifications Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-qualifications-policy",
        "Intellectual Property Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/intellectual-property-policy",
        "Program Accreditation Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/program-accreditation-policy",
    }

# Load FAISS index and all_chunks
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(ALL_CHUNKS_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(ALL_CHUNKS_PATH, "rb") as f:
        all_chunks = pickle.load(f)
    valid_policies = fetch_policies()
else:
    st.error("FAISS index is missing. Please regenerate embeddings.")
    st.stop()

# Function to get text embeddings with retry for rate limits
def get_text_embedding(text):
    retries = 5
    for attempt in range(retries):
        try:
            response = client.embeddings.create(model="mistral-embed", inputs=[text])
            return np.array(response.data[0].embedding)
        except requests.exceptions.RequestException as e:
            if "rate limit exceeded" in str(e).lower():
                wait_time = 2 ** attempt  # Exponential backoff
                st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                st.error(f"Error generating embedding: {e}")
                return None
    return None

# **Title Section**
st.title("UDST Policy Chatbot")
st.write("Ask questions about UDST policies and get relevant answers.")

st.subheader("Available Policies")
st.markdown(
    "<div style='height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px;'>"
    + "<br>".join([f'<a href="{url}" target="_blank">{title}</a>' for title, url in valid_policies.items()])
    + "</div>",
    unsafe_allow_html=True
)

# **User Query Section**
st.subheader("Ask a Question")
question = st.text_input("Enter your question:")

if question:
    if index is None or all_chunks is None:
        st.error("FAISS index is not available. Please try again later.")
    else:
        question_embedding = get_text_embedding(question)
        if question_embedding is None:
            st.error("Failed to generate embedding. Please try again.")
        else:
            # Search FAISS for top 3 related policies
            D, I = index.search(question_embedding.reshape(1, -1), k=3)
            retrieved_chunks = [all_chunks[i] for i in I.tolist()[0]]
            
            # Display top 3 policy links
            st.subheader("Top 3 Related Policies")
            for i in I.tolist()[0]:
                policy_name = list(valid_policies.keys())[i]
                policy_url = valid_policies[policy_name]
                st.markdown(f"- [{policy_name}]({policy_url})")

            # Generate response using Mistral with retry
            prompt = f"""
            Context information is below.
            ---------------------
            {retrieved_chunks}
            ---------------------
            Given the context information and not prior knowledge, answer the query.
            Query: {question}
            Answer:
            """
            messages = [UserMessage(content=prompt)]
            
            for attempt in range(5):
                try:
                    response = client.chat.complete(model="mistral-large-latest", messages=messages)
                    answer = response.choices[0].message.content if response.choices else "No response generated."
                    break
                except requests.exceptions.RequestException as e:
                    if "rate limit exceeded" in str(e).lower():
                        wait_time = 2 ** attempt
                        st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        st.error(f"Error generating response: {e}")
                        answer = "Error: Could not generate response."

            st.subheader("Answer")
            st.text_area("Answer:", answer, height=200)
