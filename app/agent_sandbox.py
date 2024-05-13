import csv
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from openai import OpenAI
import json
import dotenv  
import os 

dotenv.load_dotenv()


# Constants
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
TOKEN_JSON_FILE = 'token.json'
CLIENT_SECRETS_FILE = os.getenv("CLIENT_SECRETS_FILE")
API_KEY = os.getenv("API_KEY")
SCOPES = os.getenv("SCOPES")
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")
CSV_FILE_PATH = os.getenv("CSV_FILE_PATH")  


def load_contacts_from_csv(file_path):
    contacts = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                contacts.append({
                    'name': row['Name'],
                    'email': row['E-mail 1 - Value'],
                    'industry': row['Occupation']
                })
    except Exception as e:
        print(f"Failed to load contacts: {e}")
    return contacts

def generate_email_content(contact):
    client = OpenAI(api_key=API_KEY)
    prompt = f"""Draft a personalized email to {contact['name']} who works at Humane and is a growth marketer. She is the sender's fiancee. The sender, Nic, is 
    looking for a professional who can help launch his startup product with minimal spend. Nic knows Isabelle's dog, Koa, which is how he made the connection."""
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an expert networker who excels at writing very brief, but very powerful messages to promote the user to the recipient"},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content.strip()

def send_email(recipient_email, email_content):
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASSWORD)
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = recipient_email
        msg['Subject'] = 'Opportunity to Collaborate'
        msg.attach(MIMEText(email_content, 'plain'))
        server.send_message(msg)
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

def save_credentials():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
    creds = flow.run_local_server(port=0)
    with open(TOKEN_JSON_FILE, 'w') as token:
        token.write(creds.to_json())

def get_sent_emails():
    creds = Credentials.from_authorized_user_file('token.json', scopes=[SCOPES])
    service = build('gmail', 'v1', credentials=creds)

    results = service.users().messages().list(userId='me', labelIds=['SENT'], maxResults=100).execute()
    messages = results.get('messages', [])

    emails = []
    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
        try:
            # Extracting the body of the email from 'parts'
            body = msg['payload']['parts'][0]['body']['data']
            # Decoding from base64
            body_text = base64.urlsafe_b64decode(body).decode('utf-8')
            emails.append(body_text)
        except KeyError:
            # If there are no parts, use the snippet
            emails.append(msg['snippet'])

    return emails

def save_emails_to_json(emails, filename='emails.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(emails, f, ensure_ascii=False, indent=4)
        
def generate_contextual_prompt(emails):
    # Combine all emails into one large string to provide as context
    combined_emails = " ".join(emails)
    
    # Create a prompt for GPT-4 to generate a response based on the combined emails
    contextual_prompt = f"Based on the following email interactions: {combined_emails} Generate a contextual response that mimics the voice of the previous emails."
    
    return contextual_prompt

# Get the saved emails from JSON
def load_emails_from_json(filename='emails.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        emails = json.load(f)
    return emails


def draft_email_to_isabelle(contextual_prompt, contact):
    client = OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant trained to draft professional emails."},
            {"role": "user", "content": contextual_prompt}
        ]
    )
    email_draft = response.choices[0].message.content.strip()
    return email_draft

______
#stopped here

# def filter_emails_by_contact(emails, contact_email):
#     # Filter emails to find interactions with a specific contact
#     filtered_emails = [email for email in emails if contact_email in email]
#     return filtered_emails

# # Assuming 'contact_email' is the email of Isabelle
# isabelle_emails = filter_emails_by_contact(emails, isabelle_contact['email'])

# # Generate a new contextual prompt with filtered emails
# contextual_prompt_isabelle = generate_contextual_prompt(isabelle_emails)




#it sort of worked to generate a voice for nic, but I don't think it's that good. More work to be done. 
#After that, I'll want to somehow pull interactions between contact to keep relevant context
#next I'll want to oull data about the person being contacted so that I can personalize the message to them. 

# Main execution
if __name__ == "__main__":
    # Fetch and save emails
    emails = get_sent_emails()
    save_emails_to_json(emails)

    # Load emails and generate contextual prompt
    emails = load_emails_from_json()
    contextual_prompt = generate_contextual_prompt(emails)

    # Assuming 'contact' is defined somewhere, e.g., loaded from CSV
    contacts = load_contacts_from_csv('contacts.csv')
    isabelle_contact = next((contact for contact in contacts if contact['name'] == 'Isabelle Rossi de Leon'), None)

    if isabelle_contact:
        # Draft and print email to Isabelle
        email_to_isabelle = draft_email_to_isabelle(contextual_prompt, isabelle_contact)
        print(email_to_isabelle)
    else:
        print("Isabelle Rossi de Leon not found in contacts.")