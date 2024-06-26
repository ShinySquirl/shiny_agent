import mailbox
from email.utils import parsedate_to_datetime
import json

def get_latest_emails(mbox_path, sender_email, num_emails=100, output_file='isa_emails.json'):
    mbox = mailbox.mbox(mbox_path)
    emails = []

    for message in mbox:
        if sender_email in message['From']:
            email_data = {
                'Date': parsedate_to_datetime(message['Date']).isoformat() if message['Date'] else None,
                'From': message['From'],
                'To': message['To'],
                'Cc': message.get('Cc', ''),
                'Subject': message['Subject'],
                'Message-ID': message['Message-ID'],
                'In-Reply-To': message.get('In-Reply-To', ''),
                'References': message.get('References', ''),
                'Body': ''
            }

            # Improved handling for the email body
            if message.is_multipart():
                body_parts = []
                for part in message.walk():
                    if part.get_content_type() == 'text/plain' or part.get_content_type() == 'text/html':
                        body_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        body_parts.append(body_content)
                email_data['Body'] = ''.join(body_parts)
            else:
                email_data['Body'] = message.get_payload(decode=True).decode('utf-8', errors='ignore') if message.get_payload(decode=True) else ""

            # Handling attachments (simplified example)
            if message.is_multipart():
                attachments = []
                for part in message.walk():
                    if part.get_content_maintype() == 'multipart':
                        continue
                    if part.get('Content-Disposition') is None:
                        continue
                    filename = part.get_filename()
                    if filename:
                        attachments.append(filename)
                email_data['Attachments'] = attachments

            emails.append(email_data)

    # Sort emails by date (latest first)
    emails.sort(key=lambda x: x['Date'], reverse=True)

    # Select the latest num_emails
    latest_emails = emails[:num_emails]

    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(latest_emails, f, indent=4)

    return latest_emails

# Example usage
mbox_path = '/Users/nicbaird/Downloads/All mail Including Spam and Trash-002.j2zPI6yV.mbox.part'
sender_email = 'isabelle.rossideleon@gmail.com'
get_latest_emails(mbox_path, sender_email)