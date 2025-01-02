from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.vectordb.pgvector import PgVector, SearchType
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
from phi.model.groq import Groq
import os 
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Define the PDF Knowledge Base and the vector database setup
pdf_knowledge_base = PDFKnowledgeBase(
    path="Walmart.pdf",
    vector_db=PgVector(
        table_name="pdf_documents",
        db_url=db_url,
        embedder=SentenceTransformerEmbedder(model="FINGU-AI/FingUEm_V3"),
    ),
    reader=PDFReader(chunk=True),
)

# Define the agent with Groq model
agent = Agent(
    model=Groq(id="llama-3.1-70b-versatile", GROQ_API_KEY=os.getenv('GROQ_API_KEY')),
    knowledge=pdf_knowledge_base,
    search_knowledge=True,
    debug_mode=True, 
    instructions = [
        "Only use the Walmart pdf in the knowledge base to answer questions.",
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
# ask_agent("explain deep learning?")
# ask_agent("What is Sam's Club Segment?")
ask_agent("What is the capital of France?")  # Not in knowledge base
"""
