import os
import warnings
from dotenv import load_dotenv
from operator import itemgetter
from datetime import datetime
import requests
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough

load_dotenv()
warnings.filterwarnings("ignore")

# Groq model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# Real-time data fetcher
def get_real_time_context():
    """Fetch real-time data (weather, news, time, etc.)"""
    context = f"\n\n[Current Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
    
    try:
        # Add weather data if available
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weather_response = requests.get('https://wttr.in/?format=j1', timeout=2)
            if weather_response.status_code == 200:
                weather_data = weather_response.json()
                current_weather = weather_data['current_condition'][0]
                context += f"\n[Current Weather: {current_weather['description']}, Temp: {current_weather['temp_C']}°C]"
    except Exception:
        pass
    
    return context

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant with real-time awareness. You have access to current date/time and optional weather data. Answer all questions to the best of your ability in {language}. Use the real-time context provided to give current and accurate answers.",
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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "chat1"
if "real_time_data" not in st.session_state:
    st.session_state.real_time_data = get_real_time_context()

# Streamlit UI Configuration
st.set_page_config(
    page_title="AI ChatBot - Real-Time",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress streamlit specific warnings
import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 2rem 1rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        padding: 1.2rem;
        border-radius: 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Gradient backgrounds */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .ai-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
        color: #667eea;
        text-align: center;
    }
    
    /* Title styling */
    .title-main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        color: white;
        margin: 0.5rem 0;
    }
    
    /* Real-time status */
    .rt-status {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        color: white;
        margin: 1rem 0;
        font-weight: bold;
    }
    
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown(f'<div class="sidebar-header">⚙️ Settings</div>', unsafe_allow_html=True)
    
    language = st.selectbox(
        "🌍 Select Language:",
        ["English", "Spanish", "French", "German", "Chinese", "Japanese"],
        index=0
    )
    
    st.divider()
    
    # Real-time data display
    st.markdown("### 📡 Real-Time Data")
    rt_data = get_real_time_context()
    if "Weather" in rt_data:
        st.success("🌤️ Weather data available")
    st.info(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    
    st.divider()
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_btn"):
            st.session_state.messages = []
            store[st.session_state.session_id] = ChatMessageHistory()
            st.rerun()
    
    with col2:
        if st.button("🔄 New Chat", use_container_width=True, key="new_btn"):
            st.session_state.session_id = f"chat_{datetime.now().timestamp()}"
            st.session_state.messages = []
            st.rerun()
    
    st.divider()
    
    # Chat statistics
    st.markdown("### 📊 Chat Statistics")
    total_messages = len(st.session_state.messages)
    user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
    ai_msgs = sum(1 for m in st.session_state.messages if m["role"] == "assistant")
    
    st.metric("Total Messages", total_messages)
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("👤 User", user_msgs)
    with col_b:
        st.metric("🤖 AI", ai_msgs)
    
    st.text(f"Session: {st.session_state.session_id[:12]}...")
    
    st.divider()
    
    # Help section
    st.markdown("### 💡 Tips & Features")
    st.markdown("""
    ✨ **Features:**
    - Real-time date/time awareness
    - Weather data integration
    - Multi-language support
    - Chat history with context
    
    📝 **Commands:**
    - Type your question
    - Use 'Clear Chat' to reset
    - Use 'New Chat' for fresh session
    """)

# Main Chat Area
st.markdown("""
<div class="title-main">
    <h1>🤖 AI ChatBot with Real-Time Data</h1>
    <p>Intelligent assistant powered by Groq - Fast, Accurate, Real-Time Aware</p>
</div>
""", unsafe_allow_html=True)

# Display real-time status
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📅 Current Time", datetime.now().strftime("%H:%M:%S"))
with col2:
    st.metric("📆 Date", datetime.now().strftime("%Y-%m-%d"))
with col3:
    status = "✅ Online" if os.getenv("GROQ_API_KEY") else "❌ No API Key"
    st.metric("🔌 Status", status)

st.divider()

# Display chat history
with st.container():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])

# Input area
st.divider()

col_input, col_send = st.columns([0.95, 0.05])

with col_input:
    user_input = st.chat_input(
        "💬 Type your message here (real-time data will be included)...",
        key="user_input",
        max_chars=2000
    )

# Process user input
if user_input:
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    
    # Store in session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get real-time context
    real_time_context = get_real_time_context()
    enhanced_input = f"{user_input}{real_time_context}"
    
    # Get AI response
    config = {"configurable": {"session_id": st.session_state.session_id}}
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 Thinking with real-time data..."):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    response = chatbot.invoke(
                        {
                            "messages": [HumanMessage(content=enhanced_input)],
                            "language": language,
                        },
                        config=config,
                    )
                    ai_response = response.content
            except Exception as e:
                ai_response = f"❌ Error: {str(e)}\n\n💡 Make sure GROQ_API_KEY is set in .env file"
        
        st.markdown(ai_response)
    
    # Store AI response in session state
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #999; font-size: 0.85rem;'>
    <p>🚀 Powered by Groq (Fast AI) + Streamlit | ⏰ Real-Time Aware | 🌍 Multi-Language</p>
    <p>📡 Includes live date/time and weather data for accurate responses</p>
    </div>
    """,
    unsafe_allow_html=True
)

# To run this file:
# streamlit run AIChatBot.py
