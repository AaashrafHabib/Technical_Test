<h1 align="center" id="top">
  Technical Assesment 
</h1>

# Scalable Query Response System Using Phi Framework

## Overview

This project demonstrates the implementation of an advanced query-response system that leverages multiple knowledge sources, including PDF documents, CSV files, and external financial data. The system is designed to efficiently handle and retrieve relevant information from large datasets using vector-based search, Groq inference, and the Phi framework for agent-based development.

The solution enables the development of intelligent agents that can interact with data, process complex queries, and return accurate responses. This is useful in various domains, including financial data analysis, document-based question answering, and business intelligence.

## Features

- **PDF-Based Query Resolution**: The system can process and respond to user queries based on PDF documents (e.g., company reports).
- **CSV-Based Knowledge Retrieval**: The system is capable of answering queries from structured data in CSV files (e.g., sales data).
- **Financial Data Integration**: Real-time financial data, such as stock prices and company information, can be queried using the YFinance integration.
- **Groq Inference Engine**: Utilizes Groq for efficient model inference to accelerate the query processing.
- **PgVector Database**: Uses a vector database to store and retrieve embeddings of documents for fast similarity-based searches.

## Technologies Used

- **Python**: The core programming language used for implementing the system.
- **Phi Framework**: A powerful framework for building agent-based systems and knowledge retrieval.
- **Groq**: An inference engine used for accelerating deep learning model computations.
- **PgVector**: A PostgreSQL extension for vector similarity searches.
- **YFinance**: A tool for retrieving real-time financial data such as stock prices and news.
- **Sentence Transformers**: Used to embed the text data into vector representations for efficient retrieval.

## Architecture

The architecture of the solution follows a modular agent-based design:

1. **Agent Model**: Each agent is designed to handle specific types of queries and interact with particular knowledge sources.
2. **Knowledge Bases**:
   - **CSV Knowledge Base**: Interacts with structured data from CSV files.
   - **PDF Knowledge Base**: Interacts with unstructured data from PDF documents.
   - **Finance Agent** : Interacts with Yahoo finance to exract real time finance data . 
3. **Search Engine**: Uses **PgVector** to store and search for vectorized data, enabling fast retrieval of relevant documents.
4. **Groq Inference**: The agents use Groq for fast query processing and inference with large language models.

## Installation

To set up the environment and run the project locally, follow the steps below:

### 1. Clone the repository
```bash
git clone https://github.com/AaashrafHabib/Technical_Test
```
### 2. Run a seperate Container for Postgres to serve as a vector Database later : 
```bash
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  phidata/pgvector:16

```
### 2.Create a Python Virtual Environment  : 

#### 1.Install the required package : 

```bash
sudo apt update
sudo apt install python3-venv
```

#### 2.Create a virtual environment : 

```bash
python3 -m venv myenv

```
#### 3.Activate the virtual environment : 
```bash
source myenv/bin/activate


```

### 4.Install all the dependencies required for this project : 
```bash
Pip install -r requirements.txt

```
### 5.Get a Groq API key : 
You should register and follow this link https://console.groq.com/keys

## Agents : 
### 1.Agentic RAG: 
We were the first to pioneer Agentic RAG using our Auto-RAG paradigm. With Agentic RAG (or auto-rag), the Agent can search its knowledge base (vector db) for the specific information it needs to achieve its task, instead of always inserting the "context" into the prompt. ( in our case we used The pgVector from PostgreSQL ) 

This saves tokens and improves response quality. Create a file `agentic_rag.py`

```python
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
```
### 1.Parameters : 
  These are some per default parameters used in the agentic_Rag 
  #### Chunking strategy :
	FixedSizeChunking	is The chunking strategy to use.
  #### Reader : 
  PDFReader()	A PDFReader that converts the PDFs into Documents for the vector database.

  #### Search Type : 
  SearchType is  : vector
  Vector search involves comparing a query vector against a database of vectors to find the most similar ones

  #### Cosine for Similarity search : 
  Distance used for similarity search 	is cosine	: 
   <code> Cosine Similarity = (A · B) / (||A|| .||B||)</code>
   where <code> (A.B) </code>  : is the dot product of the two vectors. 
   <code> ( ||A|| .||B|| ) </code>  : is the  product of  the magnitudes of two vectors.

   For each vector in the database, the similarity (or distance) between the query vector and stored vectors is computed. With pgvector, cosine similarity is often used.

   Cosine Distance is defined as:
  <code> ( Cosine Distance = 1 − Cosine Similarity ) </code> 
  This converts a similarity measure (higher is better) into a distance measure (lower is better) suitable for nearest neighbor search.


### 2.CSV_Agent: 
This agent is able to read csv files and store them under a specific table on our pgvector and then understand user queries and try to respond to them using our base_knowledge 
```python


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
    path="200_Walmart.csv",
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
```
### 3.Finance Agent : 

This agent is used to retrieve data from yahoo finance platform , so this agent doesn't rely on internal ressources at all . 

```python
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

finance_agent.print_response("What is the price of walmart stock", stream=True)
```

## Achievements : 

### 1.Scalability: : 
I Ensured that the solution can handle thousands to hundreds of thousands of documents.Since the pgVector is a seperate container that can be hosted on VM on any cloud platform 

### 2.Performance : 
I ensured a very good performance regadring metrics like latency . Choose the agentic_Rag has already over traditional rag has already improved latency . but I have to mention that groq inferencing engines has been know to be quiet impressive when it comes to open source inferencing egines 

## Challenges : 
 I have to mention that I have a very limited hardware ( 8gb of Ram ) . I spent the First three days working locally on my computer , but It seemed quiet challenging especially when it comes to inference for embedding models from **Sentence_Transformer** which requires a good amount of ram , since the input desired for the **pgVector** is 1536 . So a lot of computation
 So at the end I choosed to Transfer everything to **Azure** , I created a VM  ( **Linux as OS** ) and choosed 28 Ram and 4 **CPU's**
 ## DEMO 
 ### 1.Agentic_Rag : 
 ![image](https://github.com/user-attachments/assets/6248367b-fc7d-418f-a221-2756435d2c8b)
![image](https://github.com/user-attachments/assets/62f2340a-6ae1-4355-9037-8064e8de55f5)
![image](https://github.com/user-attachments/assets/6466ba55-bc0b-452f-8832-e34520780b3e)
![image](https://github.com/user-attachments/assets/74433e62-6eb4-4647-b744-09d043853cf3)


 

 ### 2.Finance_RAG : 
 ![image](https://github.com/user-attachments/assets/e90f5aa8-338c-4ccb-bd47-cbb4190ca136)
 ![image](https://github.com/user-attachments/assets/d6051883-c914-429d-8bd9-733f4b7710d0)



 ### 3.CSV_Agent: 
 ![image](https://github.com/user-attachments/assets/67425198-1e17-426d-8b80-64785341f69d)

 ![image](https://github.com/user-attachments/assets/983b2889-142b-4405-8f1f-8e13fcb0bf6a)
 ![image](https://github.com/user-attachments/assets/98405474-1890-471d-ba67-96f0d187bc3b)
 ![image](https://github.com/user-attachments/assets/b80594d3-aee8-4927-9cfc-148c7f7845d0)



 












