import os, base64
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

def get_email_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            os.environ["GOOGLE_CREDENTIALS_PATH"], SCOPES
        )
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as w:
            w.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

def list_unread_messages(max_results=10):
    svc = get_email_service()
    resp = svc.users().messages().list(
        userId="me", q="is:unread", maxResults=max_results
    ).execute()
    return resp.get("messages", [])

def get_message_snippet(message_id):
    svc = get_email_service()
    msg = svc.users().messages().get(
        userId="me", id=message_id, format="full"
    ).execute()
    return msg["snippet"], msg["threadId"]

def get_thread(thread_id):
    svc = get_email_service()
    thread = svc.users().threads().get(
        userId="me", id=thread_id, format="full"
    ).execute()
    return [m["snippet"] for m in thread["messages"]]

def send_email(to: str, subject: str, body: str):
    svc = get_email_service()
    mime = MIMEText(body)
    mime["to"] = to
    mime["subject"] = subject
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
    return svc.users().messages().send(
        userId="me", body={"raw": raw}
    ).execute()
