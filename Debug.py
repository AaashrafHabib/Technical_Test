#Just for Debugging purposes ! 
#Printed many variables here for debug ! 
import os
from dotenv import load_dotenv
load_dotenv()
print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))