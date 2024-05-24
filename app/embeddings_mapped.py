import json
import pandas as pd
import numpy as np
from nomic import AtlasDataset, embed

# Load JSON data from file
with open('files/conversations.json', 'r') as file:
    data = json.load(file)

# Convert JSON data to a DataFrame
df = pd.DataFrame(data)

# Extract the 'title' column
titles = df['title'].tolist()

# Generate embeddings for the 'title' column
output = embed.text(
    texts=titles,
    model='nomic-embed-text-v1.5',
    task_type='search_document',
    dimensionality=512  # Optional: specify the dimensionality if needed
)

# Convert embeddings to a NumPy array
embeddings = np.array(output['embeddings'])

# Create a DataFrame with the required fields
upload_df = pd.DataFrame({
    'id': df.index,  # Use the DataFrame index as the ID field
    'title': titles
})

# Create an AtlasDataset
dataset = AtlasDataset(
    identifier="unique_conversations_dataset",
    description="Dataset of conversation titles",
    unique_id_field="id",
    is_public=True
)

# Add data to the dataset
dataset.add_data(
    data=upload_df,
    embeddings=embeddings  # Use the embeddings generated in the previous step
)

# Create an index with topic modeling
map = dataset.create_index(
    indexed_field='title',
    topic_model={'build_topic_model': True, 'topic_label_field': 'title'}
)