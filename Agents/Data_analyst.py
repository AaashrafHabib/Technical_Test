import json
from phi.model.openai import OpenAIChat
from phi.agent.duckdb import DuckDbAgent
from phi.model.groq import Groq
import os 
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


data_analyst = DuckDbAgent(
    model=Groq(id="llama-3.1-70b-versatile",GROQ_API_KEY=os.getenv('GROQ_API_KEY')),
    semantic_model=json.dumps(
        {
            "tables": [
                {
                    "name": "movies",
                    "description": "Contains information about movies from IMDB.",
                    "path": "https://phidata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv",
                }
            ]
        }
    ),
    markdown=True,
)
data_analyst.print_response(
    "Show me a histogram of ratings. ",
    stream=True,
)
