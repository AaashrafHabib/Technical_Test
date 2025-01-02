

from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.vectordb.pgvector import PgVector, SearchType
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
from phi.model.groq import Groq
import os 
from dotenv import load_dotenv
from phi.knowledge.csv import CSVReader

# Load environment variables from the .env file
load_dotenv()
from phi.knowledge.csv import CSVKnowledgeBase
from phi.vectordb.pgvector import PgVector

knowledge_base = CSVKnowledgeBase(
    path="Walmart.csv",
    # Table name: ai.csv_documents
    vector_db=PgVector(
        table_name="csv_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        embedder=SentenceTransformerEmbedder(model="FINGU-AI/FingUEm_V3")
        
    ), 
     reader=CSVReader() , 
)

# Define the agent with Groq model
agent = Agent(
    model=Groq(id="llama-3.1-70b-versatile", GROQ_API_KEY=os.getenv('GROQ_API_KEY')),
    knowledge=knowledge_base,
    search_knowledge=True,
    debug_mode=True, 
    instructions = [  
        
        "understand the query and try to answers using the  files stored in the knowledge base under the table csv_documents",
        "Do not reference any external sources or websites.",
    ],
    show_tool_calls=True,
    markdown=True,
)

# Load the knowledge base without recreating it
agent.knowledge.load(recreate=False)

"""
# Custom response handling logic
def ask_agent(question):
    response = agent.print_response(question)
    
    # Check if the response is None or contains fallback information
    if response is None or "Sorry, I cannot answer this question" in response or response.strip() == "" or "general knowledge" in response:
        print("Sorry, I cannot answer this question.")
    else:
        print(response)

# Example questions
ask_agent("what is the Temperature  in 12-02-2010 ?  " ) 
# ask_agent("What is Sam's Club Segment?")
# ask_agent("What is the capital of France?")  # Not in knowledge base

""" 