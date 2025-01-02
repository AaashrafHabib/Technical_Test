from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.model.groq import Groq
import os 
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


finance_agent = Agent(
    name="Finance Agent",
    model=Groq(id="llama-3.1-70b-versatile",GROQ_API_KEY=os.getenv('GROQ_API_KEY')),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# finance_agent.print_response("What is the price of walmart stock", stream=True)