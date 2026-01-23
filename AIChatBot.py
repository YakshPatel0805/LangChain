import os
from dotenv import load_dotenv
from operator import itemgetter
from datetime import datetime
import streamlit as st
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough

load_dotenv()

# model
from langchain_groq import ChatGroq

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

trimmer = trim_messages(
    max_tokens=45,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

chain = (
    RunnablePassthrough.assign(
        messages=itemgetter("messages") | trimmer
    )
    | prompt
    | model
)

chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "chat1"

st.set_page_config(
    page_title="AI ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem 0;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .ai-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #667eea;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f'<div class="sidebar-header">⚙️ Settings</div>', unsafe_allow_html=True)
    
    language = st.selectbox(
        "Select Language:",
        ["English", "Spanish", "French", "German", "Chinese", "Japanese"],
        index=0
    )
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            store[st.session_state.session_id] = ChatMessageHistory()
            st.rerun()
    
    with col2:
        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.session_id = f"chat_{datetime.now().timestamp()}"
            st.session_state.messages = []
            st.rerun()
    
    st.divider()
    
    st.markdown("### 📊 Chat Info")
    st.metric("Messages", len(st.session_state.messages))
    st.text(f"Session ID: {st.session_state.session_id[:8]}...")
    
    st.divider()
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Type your question or statement
    - Use 'Clear Chat' to reset conversation
    - Use 'New Chat' for a fresh session
    - Select your preferred language
    """)

st.title("🤖 AI ChatBot")
st.markdown("*Your intelligent conversational assistant powered by LLM*")

chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])

st.divider()

col1, col2 = st.columns([0.9, 0.1])

with col1:
    user_input = st.chat_input(
        "Type your message here...",
        key="user_input",
        max_chars=1000
    )

with col2:
    pass 

if user_input:
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    config = {"configurable": {"session_id": st.session_state.session_id}}
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 Thinking..."):
            response = chatbot.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "language": language,
                },
                config=config,
            )
        
        st.markdown(response.content)
    
    st.session_state.messages.append({"role": "assistant", "content": response.content})

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #999; font-size: 0.85rem;'>
    Powered by Groq (Fast AI) & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

# To run this file:
# streamlit run app_web.py
