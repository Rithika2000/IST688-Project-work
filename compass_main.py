import os
import streamlit as st
import pandas as pd
import docx
from typing import Dict
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from chromadb.config import Settings

import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3

# Set up Streamlit page configuration
st.set_page_config(
    page_title="COMPASS - University Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {}

class UniversityRecommendationSystem:
    def __init__(self):
        """Initialize the recommendation system with necessary components."""
        self.openai_api_key = st.secrets["OpenAI_key"]
        self.weather_api_key = st.secrets["open-weather"]
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Changed the data path to /tmp
        self.data_path = "/tmp/compass_data"  # Temporary storage for Streamlit Cloud
        os.makedirs(os.path.join(self.data_path, "preferences"), exist_ok=True)
        self.initialize_databases()
        self.setup_tools()
        self.setup_agent()

    def load_word_document(self, file_path: str) -> str:
        """Load content from a Word document."""
        try:
            doc = docx.Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            st.error(f"Error loading Word document: {str(e)}")
            return ""

    def initialize_databases(self):
        """Initialize ChromaDB instances with different datasets."""
        try:
            chroma_settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
            
            # Load datasets from the updated data_path
            living_expenses_df = pd.read_csv(os.path.join(self.data_path, "avglivingexpenses.csv"))
            employment_df = pd.read_csv(os.path.join(self.data_path, "Employment Projections.csv"))
            university_text = self.load_word_document(os.path.join(self.data_path, "uni100.docx"))
            
            # Process living expenses
            living_expenses_loader = DataFrameLoader(
                living_expenses_df,
                page_content_column="State"
            )
            living_expenses_docs = living_expenses_loader.load()
            
            # Process employment projections
            employment_loader = DataFrameLoader(
                employment_df,
                page_content_column="Occupation Title"
            )
            employment_docs = employment_loader.load()
            
            # Process university data
            university_docs = self.text_splitter.create_documents([university_text])
            
            # Create ChromaDB instances
            self.university_db = Chroma.from_documents(
                documents=university_docs,
                embedding=self.embeddings,
                collection_name="university_info",
                client_settings=chroma_settings
            )
            
            self.living_expenses_db = Chroma.from_documents(
                documents=living_expenses_docs,
                embedding=self.embeddings,
                collection_name="living_expenses",
                client_settings=chroma_settings
            )
            
            self.employment_db = Chroma.from_documents(
                documents=employment_docs,
                embedding=self.embeddings,
                collection_name="employment_projections",
                client_settings=chroma_settings
            )
        except Exception as e:
            st.error(f"Error initializing databases: {str(e)}")
            raise e

    def setup_tools(self):
        """Set up tools for the LangChain agent."""
        self.tools = [
            Tool(
                name="Living Expenses",
                func=self.get_living_expenses,
                description="Get information about living expenses in different states"
            ),
            Tool(
                name="Job Market Trends",
                func=self.get_job_market_trends,
                description="Get information about job market trends for different fields"
            ),
            Tool(
                name="University Information",
                func=self.get_university_info,
                description="Get information about universities and their programs"
            ),
            Tool(
                name="Weather Information",
                func=self.get_weather_info,
                description="Get current weather information for a city"
            )
        ]

    def setup_agent(self):
        """Set up the LangChain agent."""
        try:
            llm = ChatOpenAI(
                temperature=0.5,  # Reduced temperature for more focused responses
                model_name="gpt-4-turbo-preview",
                openai_api_key=self.openai_api_key,
                max_tokens=300  # Limit response length
            )

            prefix = """You are COMPASS, a concise university recommendation assistant for international students. 
            Be brief and direct in your responses while considering:
            1. Academic fit
            2. Cost and affordability
            3. Location and weather
            4. Job prospects
            
            Guidelines:
            - Keep responses under 150 words
            - Focus on most relevant information
            - Use bullet points for clarity
            - Provide specific recommendations"""

            suffix = """Begin!

            Current conversation:
            {chat_history}
            
            Human: {input}
            Assistant: Let me help you find the best matches.
            
            {agent_scratchpad}"""

            prompt = ZeroShotAgent.create_prompt(
                tools=self.tools,
                prefix=prefix,
                suffix=suffix,
                input_variables=["chat_history", "input", "agent_scratchpad"]
            )

            llm_chain = LLMChain(llm=llm, prompt=prompt)

            self.agent = ZeroShotAgent(
                llm_chain=llm_chain,
                allowed_tools=[tool.name for tool in self.tools],
                max_iterations=3  # Limit number of tool calls
            )

            self.agent_executor = AgentExecutor.from_agent_and_tools(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3,  # Limit iterations
                early_stopping_method="generate"  # Stop if stuck
            )

        except Exception as e:
            st.error(f"Error setting up agent: {str(e)}")
            raise e

    def get_living_expenses(self, state: str) -> str:
        """Retrieve living expenses information."""
        try:
            results = self.living_expenses_db.similarity_search(state, k=1)
            return results[0].page_content if results else "No information found."
        except Exception as e:
            return f"Error retrieving living expenses: {str(e)}"

    def get_job_market_trends(self, field: str) -> str:
        """Retrieve job market trends."""
        try:
            results = self.employment_db.similarity_search(field, k=3)
            return "\n".join([doc.page_content for doc in results])
        except Exception as e:
            return f"Error retrieving job market trends: {str(e)}"

    def get_university_info(self, query: str) -> str:
        """Retrieve university information."""
        try:
            results = self.university_db.similarity_search(query, k=3)
            return "\n".join([doc.page_content for doc in results])
        except Exception as e:
            return f"Error retrieving university information: {str(e)}"

    def get_weather_info(self, city: str) -> str:
        """Retrieve weather information."""
        try:
            weather_data = self.get_weather_data(city)
            if "error" in weather_data:
                return f"Could not fetch weather data: {weather_data['error']}"
            return f"Current temperature: {weather_data['main']['temp']}°F, Conditions: {weather_data['weather'][0]['description']}"
        except Exception as e:
            return f"Error retrieving weather information: {str(e)}"

    def get_recommendations(self, query: str) -> str:
        """Get personalized recommendations based on user query."""
        try:
            # Enhance query with user preferences if available
            if st.session_state.user_preferences:
                prefs = st.session_state.user_preferences
                enhanced_query = f"""Briefly answer: {query} 
                Consider preferences:
                - Field: {prefs.get('field_of_study')}
                - Budget: ${prefs.get('budget_min')}-${prefs.get('budget_max')}
                - Locations: {', '.join(prefs.get('preferred_locations', []))}
                - Weather: {prefs.get('weather_preference')}
                Keep response concise and focused."""
            else:
                enhanced_query = f"Briefly answer: {query} Keep response concise and focused."

            try:
                response = self.agent_executor.invoke(
                    {
                        "input": enhanced_query,
                        "chat_history": st.session_state.chat_history
                    }
                )
                return response["output"]
            except Exception as e:
                st.error(f"Error generating recommendation: {str(e)}")
                return ""
        except Exception as e:
            st.error(f"Error processing recommendations: {str(e)}")
            return ""

# Initialize recommendation system
compass_system = UniversityRecommendationSystem()

# Streamlit UI components
st.title("COMPASS: University Recommendation System")

# Add user input form and response display
query = st.text_input("Enter your question or preference:")

if query:
    recommendations = compass_system.get_recommendations(query)
    st.write(recommendations)
