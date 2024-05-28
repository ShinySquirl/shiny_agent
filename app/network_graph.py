import json
import base64
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import os
from pyvis.network import Network
import webbrowser
import joblib
import re
from dotenv import load_dotenv
from textblob import TextBlob
from openai import OpenAI
import umap
import plotly.express as px
import plotly.graph_objects as go
import tiktoken
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import time

# Load environment variables
load_dotenv()

CLIENT_SECRETS_FILE = os.getenv('CLIENT_SECRETS_FILE')
SCOPES = [os.getenv('SCOPES')]
TOKEN_JSON_FILE = os.getenv('TOKEN_JSON_FILE')
ENRICHED_JSON_FILE = os.getenv('ENRICHED_JSON_FILE')
EMBEDDINGS_FILE = 'umap_embeddings.pkl'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=OPENAI_API_KEY)

# nltk.download('punkt')
# nltk.download('stopwords')

# Function to clean and tokenize email content
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    tokens = word_tokenize(text.lower())  # Tokenize
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stop words
    return ' '.join(tokens)

def sanitize_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces/newlines with a single space
    return text

def count_tokens(text, encoding='gpt2'):
    enc = tiktoken.get_encoding(encoding)
    tokens = enc.encode(text)
    return len(tokens)

def divide_list_into_batches(texts, max_tokens=8000, encoding='gpt2'):
    batches = []
    current_batch = []
    current_batch_token_count = 0

    for text in texts:
        sanitized_text = sanitize_text(text)
        if not sanitized_text:
            print(f"Skipping empty or invalid text: {text}")
            continue

        token_count = count_tokens(sanitized_text, encoding)
        if token_count > max_tokens:
            print(f"Skipping text with {token_count} tokens (exceeds max tokens per batch): {text}")
            continue

        if current_batch_token_count + token_count <= max_tokens:
            current_batch.append(sanitized_text)
            current_batch_token_count += token_count
        else:
            batches.append(current_batch)
            current_batch = [sanitized_text]
            current_batch_token_count = token_count

    if current_batch:
        batches.append(current_batch)

    return batches

def extract_top_keywords(texts, n_keywords=3):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    top_keywords = []
    for i in range(X.shape[0]):
        tfidf_sorting = np.argsort(X[i].toarray()).flatten()[::-1]
        top_n = tfidf_sorting[:n_keywords]
        keywords = [terms[j] for j in top_n]
        top_keywords.append(' '.join(keywords))

    return top_keywords


def label_clusters(texts, labels):
    cluster_texts = {}
    for text, label in zip(texts, labels):
        if label not in cluster_texts:
            cluster_texts[label] = []
        cluster_texts[label].append(text)

    cluster_keywords = {}
    for label, texts in cluster_texts.items():
        keywords = extract_top_keywords(texts)
        cluster_keywords[label] = ' '.join(keywords)

    return cluster_keywords

def assign_semantic_labels(cluster_keywords):
    labels = {}
    for label, keywords in cluster_keywords.items():
        labels[label] = keywords
    return labels

def calculate_centroids(email_df, labels):
    centroids = {}
    for label in set(labels):
        cluster_points = email_df[email_df['topic'] == label]
        centroid_x = cluster_points['x'].mean()
        centroid_y = cluster_points['y'].mean()
        centroids[label] = (centroid_x, centroid_y)
    return centroids

def visualize_topics(embeddings, labels, email_df):
    umap_model = umap.UMAP(n_neighbors=min(15, len(email_df)-1), n_components=2, min_dist=0.0, metric='cosine')
    umap_embeddings = umap_model.fit_transform(embeddings)
    
    email_df = email_df.copy()  # Create a copy to avoid SettingWithCopyWarning
    email_df.loc[:, 'x'] = umap_embeddings[:, 0]
    email_df.loc[:, 'y'] = umap_embeddings[:, 1]
    email_df.loc[:, 'topic'] = labels
    
    email_df.loc[:, 'top_keywords'] = extract_top_keywords(email_df['body'])

    cluster_keywords = label_clusters(email_df['body'], labels)
    semantic_labels = assign_semantic_labels(cluster_keywords)
    email_df.loc[:, 'semantic_topic'] = email_df['topic'].apply(lambda x: semantic_labels[x])
    
    # Calculate centroids for each cluster
    centroids = calculate_centroids(email_df, labels)
    
    fig = px.scatter(
        email_df, x='x', y='y', color='topic',
        custom_data=['sender', 'receivers', 'subject', 'top_keywords', 'semantic_topic'],
        title='UMAP projection of the email embeddings'
    )
    fig.update_traces(
    hovertemplate="<br>".join([
        "Sender: %{customdata[0]}",
        "Receivers: %{customdata[1]}",
        "Subject: %{customdata[2]}",
        "Top Keywords: %{customdata[4]}",
        "Semantic Topic: %{customdata[5]}"
    ])
    )
    
    fig.update_layout(
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        legend_title='Topic'
    )
    
    # Add text annotations for semantic topics
    for label, (x, y) in centroids.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            text=[semantic_labels[label]],
            mode='text',
            showlegend=False
        ))
    
    fig.show()
    
    return umap_embeddings

# Function to get OpenAI embeddings
def get_openai_embeddings(texts):
    embeddings = []
    sanitized_texts = [sanitize_text(text) for text in texts if sanitize_text(text)]
    skipped_texts = []

    start_time = time.time()
    batches = divide_list_into_batches(sanitized_texts, max_tokens=8000)
    total_batches = len(batches)

    for batch_num, batch in enumerate(batches):
        print(f"Processing batch {batch_num + 1}/{total_batches}...")
        print(f"Batch content: {json.dumps(batch, indent=2)}")

        try:
            response = client.embeddings.create(input=batch, model="text-embedding-3-small")
            embeddings.extend(np.array(item.embedding) for item in response.data)
        except Exception as e:
            print(f"Error processing batch {batch_num + 1}: {e}")
            skipped_texts.extend(batch)
            continue

    end_time = time.time()
    print(f"Total time taken for generating embeddings: {end_time - start_time} seconds")
    print(f"Total number of embeddings generated: {len(embeddings)}")
    print(f"Skipped texts: {skipped_texts}")

    return embeddings, skipped_texts

def generate_or_load_embeddings(texts, force_regenerate=False):
    if os.path.exists(EMBEDDINGS_FILE) and not force_regenerate:
        print("Loading existing embeddings...")
        umap_embeddings = joblib.load(EMBEDDINGS_FILE)
    else:
        print("Generating new embeddings...")
        embeddings, skipped_texts = get_openai_embeddings(texts)

        if len(embeddings) != len(texts):
            print(f"Warning: Number of embeddings ({len(embeddings)}) does not match number of texts ({len(texts)})")
            print(f"Skipped texts: {skipped_texts}")

        umap_model = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='cosine')
        umap_embeddings = umap_model.fit_transform(embeddings)
        joblib.dump(umap_embeddings, EMBEDDINGS_FILE)

    return umap_embeddings

def create_network_graph(email_df, umap_embeddings):
    net = Network(height='750px', width='100%', notebook=True)
    
    if isinstance(umap_embeddings, np.ndarray):
        umap_embeddings = umap_embeddings.tolist()
    
    for i, row in email_df.iterrows():
        x, y = float(umap_embeddings[i][0]), float(umap_embeddings[i][1])
        net.add_node(i, label=row['subject'], title=f"Sender: {row['sender']}\nReceivers: {', '.join(row['receivers'])}\nBody: {row['body']}", x=x, y=y)
    
    for i, row in email_df.iterrows():
        for receiver in row['receivers']:
            receiver_index = email_df[email_df['sender'] == receiver].index
            if not receiver_index.empty:
                net.add_edge(i, receiver_index[0])
    
    net.show('email_network.html')
    webbrowser.open('email_network.html')
    
def main():
    # Load email data
    email_data = []
    with open(ENRICHED_JSON_FILE, 'r') as f:
        for line in f:
            try:
                email = json.loads(line)
                email_data.append(email)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

    email_df = pd.DataFrame(email_data)
    email_df['body'] = email_df['body'].apply(preprocess_text)

    print(f"Number of texts before preprocessing: {len(email_data)}")
    print(f"Number of texts after preprocessing: {len(email_df['body'].tolist())}")

    sample_size = 10
    email_df_sample = email_df.head(sample_size)

    print("Texts before preprocessing:")
    for email in email_data[:sample_size]:
        print(email['body'])

    print("Texts after preprocessing:")
    for text in email_df_sample['body'].tolist():
        print(text)

    texts = email_df_sample['body'].tolist()
    print(f"Number of texts to generate embeddings for: {len(texts)}")
    umap_embeddings = generate_or_load_embeddings(texts, force_regenerate=True)

    if len(umap_embeddings) != len(texts):
        raise ValueError(f"Number of embeddings ({len(umap_embeddings)}) does not match number of texts ({len(texts)})")

    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(umap_embeddings)

    visualize_topics(umap_embeddings, labels, email_df_sample)
    create_network_graph(email_df_sample, umap_embeddings)

if __name__ == "__main__":
    main()