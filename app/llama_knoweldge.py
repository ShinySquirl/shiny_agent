import os
import json
import logging
import requests
import time
from pydantic import BaseModel
from typing import Optional, List
from tqdm import tqdm
from llama_index.core import SimpleDirectoryReader, Document, PropertyGraphIndex, StorageContext, load_index_from_storage
from llama_index.core.indices.property_graph import PropertyGraphIndex, SchemaLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPGStore
from typing import Literal
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Any, Optional
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
import asyncio
import httpx  # Import httpx for HTTP operations
from tenacity import retry, stop_after_attempt, wait_random_exponential  # Import tenacity for retries

tokenizer = Anthropic().tokenizer
Settings.tokenizer = tokenizer

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fetch environment variables
username_neo4j = os.getenv("USERNAME_NEO4J")
password_neo4j = os.getenv("PASSWORD_NEO4J")
uri_neo4j = os.getenv("URI_NEO4J")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


embed_model = OpenAIEmbedding(model_name="text-embedding-3-small", embed_batch_size=42)

llm = OpenAI(model="gpt-4-turbo", temperature=0.3)
# Define entity and relationship types
entities = Literal["PERSON", "TOPIC"]
relations = Literal["EXPERT_IN", "WORKING_ON", "WORKED_WITH", "KNOWS"]

# Define schema for valid relationships
schema = {
    "TOPIC": ["EXPERT_IN, WORKING_ON"],
    "PERSON": ["WORKED_WITH", "KNOWS", "EXPERT_IN, WORKING_ON"],
}

# Create schema extractor
kg_extractor = SchemaLLMPathExtractor(
  llm=llm,
  possible_entities=entities,
  possible_relations=relations,
  kg_validation_schema=schema,
  strict=True
  #consider trying strict=false
)


""" Functions Below """
# Function to test Neo4j connection
def test_neo4j_connection(uri, username, password):
    from neo4j import GraphDatabase
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        driver.close()
        logger.info("Successfully connected to Neo4j Aura")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        raise

# Initialize Neo4j graph store
def initialize_neo4j_store(username, password, uri):
    try:
        graph_store = Neo4jPGStore(
            username=username,
            password=password,
            url=uri
        )
        logger.info("Successfully connected to Neo4j")
        return graph_store
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        raise

# Function to initialize and save the index
def initialize_and_save_index(documents, kg_extractor, graph_store, persist_dir):
    index = PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=[kg_extractor],
        property_graph_store=graph_store,
        use_async=False
    )
    index.storage_context.persist(persist_dir=persist_dir)
    return index

# Function to load emails from JSON
def load_emails(file_path, max_emails=10):
    with open(file_path, 'r') as file:
        emails_data = json.load(file)
    
    emails = []
    for count, data in enumerate(emails_data):
        if count >= max_emails:
            break
        email_text = data['Body']
        document = Document(text=email_text)  # Only store text, no embedding here
        emails.append(document)
    return emails

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def send_request_with_throttling(url, headers, data):
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

# Main processing logic
def main():
    # Configuration
    persist_dir = "./storage"
    isa_emails_file = "isa_emails.json"
    nic_emails_file = "nic_emails.json"
    max_emails = 10

    # Step 1: Test Neo4j connection
    test_neo4j_connection(uri_neo4j, username_neo4j, password_neo4j)

    # Step 2: Initialize Neo4j graph store
    graph_store = initialize_neo4j_store(username_neo4j, password_neo4j, uri_neo4j)

    # Step 3: Load emails in batches from both files
    isa_emails = load_emails(isa_emails_file, max_emails)
    nic_emails = load_emails(nic_emails_file, max_emails)

    # Combine emails from both files
    all_emails = isa_emails + nic_emails

    # Initialize and save the index if not already done
    if not os.path.exists(persist_dir):
        initialize_and_save_index(all_emails, kg_extractor, graph_store, persist_dir)

    # Load the previously saved index
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=persist_dir)
    )

    # Create a retriever from the index
    retriever = index.as_retriever()

    # # Example of using the retriever
    # for query in ["Example query 1", "Example query 2"]:
    #     nodes = retriever.retrieve(query)
    #     print(f"Results for '{query}': {nodes}")

    # # Optional: Insert new documents if needed
    # new_emails = load_emails("new_emails.json", max_emails)
    # if new_emails:
    #     index.insert(new_emails)

if __name__ == "__main__":
    main()
