import os
import json
import logging
from pydantic import BaseModel
from typing import Optional, List
from tqdm import tqdm
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.indices.property_graph import PropertyGraphIndex, SchemaLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPGStore
import nest_asyncio
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

# Load environment variables
load_dotenv()

class Entities(BaseModel):
    """List of named entities int he text such as names of people, orgs, concepts, locations"""
    names: Optional[List[str]]
    
prompt_template_entities = """
Extract all names entities such as names of people, organizations, concepts, and locations
from the following text: 
{text}
"""


# Apply nest_asyncio for nested async loops
nest_asyncio.apply()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fetch environment variables
username_neo4j = os.getenv("USERNAME_NEO4J")
password_neo4j = os.getenv("PASSWORD_NEO4J")
uri_neo4j = os.getenv("URI_NEO4J")

llm = OpenAI(model="gpt-4o", temperature=0.3)
embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
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

# Function to load pre-embedded emails
def load_preembedded_emails(file_path, batch_size=50, max_emails=50):
    with open(file_path, 'r') as file:
        email_count = 0
        while email_count < max_emails:
            documents = []
            for _ in range(batch_size):
                if email_count >= max_emails:
                    break
                line = file.readline()
                if not line:
                    break
                data = json.loads(line.strip())
                email_text = data['body']
                embedding = data['embeddings']
                document = Document(text=email_text, embedding=embedding)
                documents.append(document)
                email_count += 1
            if not documents:
                break
            yield documents

# Function to process email batch
def process_batch(email_batch, kg_extractor, graph_store):
    index = PropertyGraphIndex.from_documents(
        email_batch,
        kg_extractors=[kg_extractor],
        llm=llm,
        embed_model=embed_model,
        property_graph_store=graph_store,
        show_progress=True
    )
    return index

# Main processing logic
def main():
    preembedded_emails_file = "files/enriched_email_interactions.json"
    batch_size = 50
    max_emails = 50
    test_neo4j_connection(uri_neo4j, username_neo4j, password_neo4j)
    graph_store = initialize_neo4j_store(username_neo4j, password_neo4j, uri_neo4j)
    email_batches = load_preembedded_emails(preembedded_emails_file, batch_size, max_emails)

    for email_batch in tqdm(email_batches):
        try:
            # Process each batch and get an index
            index = process_batch(email_batch, kg_extractor, graph_store)
            
            # Set up the retrievers using the index
            llm_synonym = LLMSynonymRetriever(
                index.property_graph_store,
                llm=llm,
                include_text=False,
            )

            vector_context = VectorContextRetriever(
                index.property_graph_store,
                embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
                include_text=False,
            )

            # Set up a retriever to query the graph
            retriever = index.as_retriever(
                sub_retrievers=[
                    llm_synonym,
                    vector_context,
                ],
                include_text=True
            )
            logger.info("Successfully set up retriever")

            # Example query
            nodes = retriever.retrieve("What happened at Interleaf?")
            for node in nodes:
                print(node.text)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")

if __name__ == "__main__":
    main()
