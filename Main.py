from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from Agents.Agentic_Rag import agent as agentic_rag 
from Agents.Finance_Agent import finance_agent
from Agents.CSV_Agent import agent as CSV_Agent 
from phi.vectordb.pgvector import PgVector, SearchType
import os 
from dotenv import load_dotenv
from phi.playground import Playground, serve_playground_app
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder




# Load environment variables from the .env file
load_dotenv()

agent_team = Agent ( 
    Name="Agent Team", 
    agent_id ="agent_team", 
    team= [finance_agent,agentic_rag,CSV_Agent,], 
    model =Groq(id="llama-3.1-70b-versatile",GROQ_API_KEY=os.getenv('GROQ_API_KEY')),
    # storage=PgVector(
    #     table_name="Agents_sessions",
    #     db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    #     embedder=SentenceTransformerEmbedder(model="FINGU-AI/FingUEm_V3")
        
    # ),
    debug_mode=True, 
    description="You are a highly advanced AI agent with access to an extensive knowledge base and powerful Finance-search capabilities.",
    # instructions=[
    #         "Always search your knowledge base first.\n"
    #         "  - Search your knowledge base before seeking external information.\n"
    #         "  - Provide answers based on your existing knowledge whenever possible.",
    #         "Then search the web if no information is found in your knowledge base.\n"
    #         "  - If the information is not available in your knowledge base, use `duckduckgo_search` to find relevant information.",
    #         "Provide concise and relevant answers.\n"
    #         "  - Keep your responses brief and to the point.\n"
    #         "  - Focus on delivering the most pertinent information without unnecessary detail.",
    #         "Ask clarifying questions.\n"
    #         "  - If a user's request is unclear or incomplete, ask specific questions to obtain the necessary details.\n"
    #         "  - Ensure you fully understand the inquiry before formulating a response.",
    #         "Verify the information you provide for accuracy.",
    #         "Cite reliable sources when referencing external data.",
    #     ],
    #all-mpnet-base-v2	The name of the SentenceTransformers model to use per default 
    markdown=True, 

)

app=Playground(agents=[finance_agent,agentic_rag,CSV_Agent, agent_team]).get_app()

if __name__=="__main__":
    serve_playground_app("Main:app",reload=True)