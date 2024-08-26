import os
import pandas as pd
from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
import mysql.connector

app = Flask(__name__)

# Load your Groq API key
GROQ_API_KEY = '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'
client = Groq(api_key=GROQ_API_KEY)

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

csv_file_path = "kpi.csv"  # Path to your CSV file
csv_loader = CSVLoader(file_path=csv_file_path)
documents = csv_loader.load()
print(f"Number of documents loaded from CSV: {len(documents)}")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
split_documents = text_splitter.split_documents(documents)

# Generate embeddings for the CSV data
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# vectorstore = Chroma.from_documents(
#     documents=split_documents, 
#     embedding=embedding_function, 
#     persist_directory="embeddings"
# )
# vectorstore.persist()

# Initialize Chroma vectorstore with embeddings
vectorstore = Chroma(
    persist_directory="embeddings",
    embedding_function=embedding_function
)
print(f"Number of documents in vectorstore: {vectorstore._collection.count()}")
# SQL Database connection details
db_config = {
    'user': 'root',
    'password': 'S#uhas@123',
    'host': 'localhost',  
    'port': 3306, 
    'database': 'kpis_db',
}

# Function to execute the SQL query and retrieve the row from the SQL database
def execute_sql_query(query):

    connection = mysql.connector.connect(**db_config)
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Execute the query
        cursor.execute(query)
        
        # Fetch the result
        result = cursor.fetchall()
        return result

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

async def rag(query: str, contexts: list) -> str:
    print("> RAG Called")
    context_str = "\n".join(contexts)
    print(context_str)
    
    # Create the prompt for the language model
    prompt = f"""
    You have been provided with a SQL table named kpis which has the following structure:

    CREATE TABLE kpis (
    KPI VARCHAR(100),
    DESCRIPTION TEXT,
    TABLE_NAME VARCHAR(100),
    COMBINED_DESCRIPTION TEXT
    );

    Your task is to generate a precise SQL query based on a given user query and a set of combined_description fields retrieved from a vector store. The SQL query should retrieve the relevant row(s) from the kpis table.

    Instructions:
    1) Analyze the User Query and Contexts: You will be given a user query and a list of combined_description fields from the kpis table. Your goal is to identify the most relevant combined_description that closely matches the user query.
    2) Select the Correct Row: Based on the best-matched combined_description, generate a SQL query to retrieve the corresponding row(s) from the kpis table. Ensure that the COMBINED_DESCRIPTION field in the SQL query matches the best-matched description.
    3) Choose the Appropriate Table: The TABLE_NAME within the combined_description might indicate different tables (e.g., Instant, Summarized, Weekly). Use this information to ensure that the SQL query is specific to the correct context.
    4) Output Only the SQL Query: The final output should be a valid SQL query that can be executed directly on the kpis table. The output should not include any additional explanations or text.

    Example:
    If a combined_description context is:

    TABLE: Summarized
    COMBINED_DESCRIPTION: Total outgoing local calls made to external networks over the past 30 days, measured in minutes.

    and this is identified as the best match for the user query, the output should be:

    SELECT * FROM kpis WHERE COMBINED_DESCRIPTION = 'Total outgoing local calls made to external networks over the past 30 days, measured in minutes.';

    Generate the SQL query based on the provided context.

    Contexts:
    {context_str}

    Query:
    {query}

    Output:
    SELECT * FROM kpis WHERE COMBINED_DESCRIPTION = '...';
    """

    # Generate answer using Groq
    llm = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a highly skilled data retrieval system designed to extract specific information from a dataset stored in an SQL database. you are the only system which give SQL query as answer"},
            {"role": "user", "content": prompt}
        ],
        model="llama3-70b-8192",
        temperature=1,
        max_tokens= 200
    )
    response = llm.choices[0].message.content.strip()
    print(response)
    return response

async def retrieve(query: str) -> list:
    print("hi")
    # Create query embedding
    print(query)
    print(f"Number of documents in vectorstore: {vectorstore._collection.count()}")
    # Perform similarity search in the vector store
    results = vectorstore.similarity_search(query, k=5)
    print(results)
    # Extract the combined descriptions from the results
    contexts = [result.page_content for result in results]
    return contexts

@app.route('/smsbot', methods=['POST'])
async def send_sms():
    received_message = request.json.get('payload', '')
    print("The received message is", received_message)

    # Execute the RAG process
    contexts = await retrieve(received_message)
    sql_query = await rag(received_message, contexts)
    
    # Execute the SQL query and fetch the result
    result = execute_sql_query(sql_query)
    
    return jsonify({"message": result})

if __name__ == "__main__":
    app.run(debug=True, port=8000)
