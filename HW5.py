import openai
import streamlit as st
import os
import glob
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from bs4 import BeautifulSoup

# Initialize the OpenAI client
if 'openai_client' not in st.session_state:
    api_key = st.secrets["OpenAI_key"]
    openai.api_key = api_key
    st.session_state.openai_client = openai

# Initialize the ChromaDB client
client = chromadb.PersistentClient(path="su_orgs")  # Change to relative path

# Initialize the Anthropic client
if 'anthropic_client' not in st.session_state:
    anthropic_api_key = st.secrets["Anthropic_key"]
    from anthropic import Anthropic
    st.session_state.anthropic_client = Anthropic(api_key=anthropic_api_key)

# Initialize the Cohere client
if 'cohere_client' not in st.session_state:
    cohere_api_key = st.secrets["Cohere_key"]
    import cohere
    st.session_state.cohere_client = cohere.Client(api_key=cohere_api_key)

# Function to create ChromaDB collection and store in session state
def create_chromadb_collection():
    if 'Lab4_vectorDB' not in st.session_state:
        collection_name = "HW4Collection"
        existing_collections = client.list_collections()
        
        if collection_name in [col.name for col in existing_collections]:
            collection = client.get_collection(collection_name)
        else:
            collection = client.create_collection(collection_name)

        # Check if the vector database already exists
        vector_db_path = "su_orgs/vector_db_exists.txt"  # Change to relative path
        if not os.path.exists(vector_db_path):
            def html_to_text(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file, 'html.parser')
                    return soup.get_text()

            def add_to_collection(collection, text, filename):
                response = st.session_state.openai_client.embeddings.create(
                    input=[text],
                    model='text-embedding-3-small'
                )
                embedding_vector = response.data[0].embedding

                collection.add(
                    documents=[text],
                    ids=[filename],
                    embeddings=[embedding_vector],
                    metadatas=[{'filename': filename}]
                )

            html_folder = "su_orgs"  # Change to relative path
            for file_name in glob.glob(os.path.join(html_folder, "*.html")):
                text_content = html_to_text(file_name)
                add_to_collection(collection, os.path.basename(file_name), text_content)

            # Create a file to indicate that the vector database has been created
            with open(vector_db_path, 'w') as f:
                f.write("Vector DB created.")

            st.session_state.Lab4_vectorDB = collection
            print(f"Collection '{collection_name}' is ready.")
        else:
            st.session_state.Lab4_vectorDB = collection
            print("Vector DB already exists. Loading the existing collection.")

# Function to get relevant course information
def get_relevant_course_info(query):
    response = st.session_state.openai_client.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding

    results = st.session_state.Lab4_vectorDB.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    relevant_texts = [result for result in results['documents'][0]]
    return "\n\n".join(relevant_texts)

# Function to query the vector DB and invoke LLM without needing system prompt construction
def query_and_chat(query):
    relevant_course_info = get_relevant_course_info(query)

    # Call the selected LLM directly with the vector search results
    if model_choice == "OpenAI":
        response = st.session_state.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "Relevant course info:\n" + relevant_course_info},
                      {"role": "user", "content": query}]
        )
        answer = response.choices[0].message.content

    elif model_choice == "Anthropic":
        response = st.session_state.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            system="Relevant course info:\n" + relevant_course_info,  # Pass system message here
            messages=[{"role": "user", "content": query}]
        )
        answer = response.content if hasattr(response, 'content') else "No response from Claude."

    elif model_choice == "Cohere":
        response = st.session_state.cohere_client.chat(
            model="command-r-plus",
            message=relevant_course_info + "\n\nUser question: " + query
        )
        answer = response.text

    return answer

# Function to limit the conversation memory
def limit_memory(messages, limit=5):
    if len(messages) > limit:
        return messages[-limit:]  # Keep only the last 'limit' messages
    return messages

# Streamlit chat interface using chat_message for a better conversation view
st.title("Course Information Chatbot")

# Sidebar for model selection
model_choice = st.sidebar.selectbox("Select LLM Model", ("OpenAI", "Anthropic", "Cohere"))

# Initialize message history in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

create_chromadb_collection()

# Chat input box
user_input = st.text_input("Ask a question about the course:", key="input")

if st.button("Send", key="send"):
    if user_input:
        answer = query_and_chat(user_input)

        # Append user and bot messages to the session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "bot", "content": answer})

        # Limit conversation memory to the last 5 exchanges
        st.session_state.messages = limit_memory(st.session_state.messages)

# Display chat history using the chat_message format
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])