from langchain_community.llms import Ollama
import dotenv
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchRun
from gmail_service import list_unread_messages
from gmail_service import get_message_snippet
from gmail_service import get_thread
from gmail_service import send_email
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain.prompts import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)

load_dotenv()

# Define memory for remembering previous conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# System prompt, sets the "jarvis" mood 
system = SystemMessagePromptTemplate.from_template(
    """
    You are an advanced AI assistant modeled after JARVIS from the movie "Iron Man." Your main function is to act as my personal agent, capable of managing my tasks and executing my commands based on the comprehensive knowledge you possess about my preferences, schedule, and information. Your abilities include managing emails, accessing my calendar, and integrating with my social media. 
    You also have the capability of seraching on the web for things that are real-time, such as whether and any news that might be happening right now. 

    Your task is to respond to my commands and queries as if you are fully aware of my activities and schedules. Here are some of my details you’ll consider when executing tasks:

    Name: Alex Wang
    Email Address: wangalex0410@gmail.com
    Calendar Schedule: __________
    Social Media Accounts Details: __________
    Please take into account my preferences, important contacts, and any recurring tasks or appointments that shape my daily routine. Make sure to maintain a conversational tone, similar to a personal assistant, and ensure your responses are efficient and relevant to my requests.

    For example, if I ask about my schedule, respond with a summary of my upcoming appointments. If I request to send an email, draft the message based on my communication style, and if i request general knowlegde that is outside the scope of my 
    personal information, search the web for any relevant information. 
"""
)

# Enables real-time online searching
search = DuckDuckGoSearchRun()

# Uses tools
def draft_reply(to: str, subject: str, thread_context: str) -> str:
    messages = prompt.format_messages(
        user_input=thread_context + f"\nReply to {to} with subject {subject}."
    )
    return chat.invoke(messages).content

class DraftReplyInput(BaseModel):
    to: str = Field(..., description="Recipient email address")
    subject: str = Field(..., description="Email subject line")
    thread_context: str = Field(..., description="Previous conversation")

draft_reply_tool = StructuredTool.from_function(
    func=draft_reply,
    name="draft_reply",
    description="Draft a professional email reply.",
    args_schema=DraftReplyInput,)

# Tool calling function for api calling
tools = [
        Tool(
        name="list_unread",
        func=lambda _: list_unread_messages(),
        description="Return a list of your unread Gmail message IDs."
    ),
    Tool(
        name="DuckDuckGo Search",
        func=lambda q: search.run(q)[:1000],
        description="This is useful for seraching the internet for any real-time information (e.g. news, stocks, etc), limits at 1000 characters"
    ),

    Tool(
        name="read_snippet",
        func=lambda msg_id: get_message_snippet(msg_id)[0],
        description="Given a Gmail message ID, return its snippet and threadId."
    ),
    Tool(
        name="fetch_thread",
        func=lambda thread_id: "\n".join(get_thread(thread_id)),
        description="Given a Gmail thread ID, return the full conversation as text."
    ),
    Tool(
        name="draft_reply",
        func=lambda args: draft_reply(**args),
        description=(
            "Draft an email reply based on context. "
            "Args is a dict 'to', 'subject', 'thread_context'."
        )
    ),
    Tool(
        name="send_email",
        func=lambda args: send_email(**args),
        description="Send an email. Args is a dict 'to', 'subject', 'body'."),

    draft_reply_tool,  
    Tool(
        name="send_email",
        func=lambda d: send_email(**d),
        description="Send an email. Expects keys: 'to', 'subject', 'body'.",
    ),
]

# Initializes human input into the llm taken from the user_input from main function, then sends the prompt to the llm in template form
human = HumanMessagePromptTemplate.from_template("{user_input}")
prompt = ChatPromptTemplate.from_messages([system, human])

chat = Ollama(
    model="phi3:mini", 
    base_url="http://localhost:11434",
    temperature=0.7,
)

agent = initialize_agent(
    tools,
    chat,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    max_execution_time=60,
    max_iterations=4,
    early_stopping_method="generate"
)

if __name__ == "__main__":
    print(
        "HI, my name is Jarvis, your personal assistant that can help you read emails, "
        "draft replies, read social-media notifications, and manage your GitHub workflow.",
        flush=True,
    )
    prompt = True
    print("How may I assist you today?", flush=True)
    while prompt:
        user_input = input("(Type exit/quit to exit)> ")

        if user_input.upper() in ("EXIT", "QUIT"):
            print("Goodbye sir.", flush=True)
            prompt = False
        else:
            response = agent.run(user_input)
            print("Jarvis:", response, flush=True)