import json
import base64
import pandas as pd
import numpy as np
from nomic import AtlasDataset, embed
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from dotenv import load_dotenv
import os

load_dotenv()

CLIENT_SECRETS_FILE = os.getenv('CLIENT_SECRETS_FILE')
SCOPES = [os.getenv('SCOPES')]
TOKEN_JSON_FILE = os.getenv('TOKEN_JSON_FILE')
ENRICHED_JSON_FILE = os.getenv('ENRICHED_JSON_FILE')

# Function to get email interactions from Gmail API
def get_email_interactions():
    creds = None
    if os.path.exists(TOKEN_JSON_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_JSON_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_JSON_FILE, 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)
    
    # Query to get all sent emails
    query = 'in:sent'
    email_data = []
    next_page_token = None

    while True:
        results = service.users().messages().list(userId='me', q=query, pageToken=next_page_token).execute()
        messages = results.get('messages', [])
        next_page_token = results.get('nextPageToken')

        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            headers = msg['payload']['headers']
            subject = next((header['value'] for header in headers if header['name'] == 'Subject'), '')
            sender = next((header['value'] for header in headers if header['name'] == 'From'), '')
            receivers = [header['value'] for header in headers if header['name'] == 'To']
            body = ''
            if 'parts' in msg['payload']:
                for part in msg['payload']['parts']:
                    if part['mimeType'] == 'text/plain':
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        break
            
            email_data.append({
                'id': msg['id'],
                'subject': subject,
                'body': body,
                'sender': sender,
                'receivers': ', '.join(receivers)  # Convert list to a single string
            })

        if not next_page_token:
            break
    
    return email_data

# Check if the enriched JSON file already exists
if os.path.exists(ENRICHED_JSON_FILE):
    # Load the enriched email interactions JSON file
    data = []
    with open(ENRICHED_JSON_FILE, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    # Convert JSON data to a DataFrame
    email_df = pd.DataFrame(data)
    # Extract embeddings from the DataFrame
    embeddings_array = np.array(email_df['embeddings'].tolist())
else:
    # Get email interactions using Gmail API
    email_data = get_email_interactions()

    # Convert email interactions data to a DataFrame
    email_df = pd.DataFrame(email_data)

    # Generate embeddings for the email subjects and bodies
    email_texts = email_df['subject'] + ' ' + email_df['body']
    embeddings = embed.text(
        texts=email_texts.tolist(),
        model='nomic-embed-text-v1.5',
        task_type='search_document',
        dimensionality=512
    )

    # Convert embeddings to a NumPy array
    embeddings_array = np.array(embeddings['embeddings'])

    # Add embeddings to the DataFrame
    email_df['embeddings'] = list(embeddings_array)

    # Save the enriched email interactions to a new JSON file
    email_df.to_json(ENRICHED_JSON_FILE, orient='records', lines=True)

# Load the enriched email interactions JSON file
data = []
with open('files/enriched_email_interactions.json', 'r') as file:
    for line in file:
        data.append(json.loads(line))

# Convert JSON data to a DataFrame
df = pd.DataFrame(data)

# Create a DataFrame with the required fields
upload_df = pd.DataFrame({
    'id': df.index,  # Use the DataFrame index as the ID field
    'subject': df['subject'],
    'body': df['body'],
    'sender': df['sender'],
    'sender': df['sender'],
    'receivers': df['receivers']
})

# Create an AtlasDataset
dataset = AtlasDataset(
    identifier="unique_email_dataset",
    description="Dataset of email subjects and bodies",
    unique_id_field="id",
    is_public=True
)

# Add data to the dataset
dataset.add_data(
    data=upload_df,
    embeddings=embeddings_array  # Use the embeddings generated in the previous step
)

# Create an index with topic modeling
map = dataset.create_index(
    indexed_field='subject',
    topic_model={'build_topic_model': True, 'topic_label_field': 'subject'}
)