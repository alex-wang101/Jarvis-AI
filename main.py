from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
import os 
import dotenv
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from gmail_service import (list_unread_messages, get_message_snippet, get_thread, send_email,)

load_dotenv()

system = SystemMessagePromptTemplate.from_template(
    """
You are an advanced AI assistant modeled after JARVIS from the movie "Iron Man." Your main function is to act as my personal agent, capable of managing my tasks and executing my commands based on the comprehensive knowledge you possess about my preferences, schedule, and information. Your abilities include managing emails, accessing my calendar, and integrating with my social media.

Your task is to respond to my commands and queries as if you are fully aware of my activities and schedules. Here are some of my details you’ll consider when executing tasks:

Name: __________
Email Address: __________
Calendar Schedule: __________
Social Media Accounts Details: __________
Please take into account my preferences, important contacts, and any recurring tasks or appointments that shape my daily routine. Make sure to maintain a conversational tone, similar to a personal assistant, and ensure your responses are efficient and relevant to my requests.

For example, if I ask about my schedule, respond with a summary of my upcoming appointments. If I request to send an email, draft the message based on my communication style.
"""
)
human = HumanMessagePromptTemplate.from_template("{user_input}")
prompt = ChatPromptTemplate.from_messages([system, human])

chat = ChatOpenAI(
    model="llama3.2", 
    base_url = "http://localhost:11434/v1",
    openai_api_key = "using ollama instead",
    temperature=0.7
)

while True:
    text = input(">")
    if text.upper() in ("EXIT", "QUIT"):
        print("Goodbye sir.")
        break

    messages = prompt.format_messages(user_input=text)
    response = chat.invoke(messages)

    print("Jarvis: ", response.content)
