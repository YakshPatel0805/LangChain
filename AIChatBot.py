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
import base64
from PIL import Image
import io
import json

# Try to import OpenAI as backup for vision
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import Google Gemini as another backup for vision
try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

load_dotenv()
warnings.filterwarnings("ignore")

# Groq models - including vision model
text_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# Updated to use the current supported vision model
vision_model = ChatGroq(
    model="llama-3.2-90b-vision-preview",  # Updated to current supported model
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# Initialize OpenAI client as backup
openai_client = None
if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Google Gemini as another backup
google_model = None
if GOOGLE_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    google_model = genai.GenerativeModel('gemini-1.5-flash')

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

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def resize_image(image, max_size=1024):
    """Resize image while maintaining aspect ratio"""
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def process_image_with_google(image, user_prompt, language="English"):
    """Process image with Google Gemini Vision"""
    if not google_model:
        return None
    
    try:
        # Resize image for optimal processing
        processed_image = resize_image(image)
        
        # Create prompt for Gemini
        prompt = f"Analyze this image and respond in {language}. User question: {user_prompt}"
        
        # Generate response with image
        response = google_model.generate_content([prompt, processed_image])
        return response.text
        
    except Exception as e:
        return f"Google Gemini Vision Error: {str(e)}"

def process_image_with_openai(image, user_prompt, language="English"):
    """Fallback: Process image with OpenAI Vision API"""
    if not openai_client:
        return None
    
    try:
        # Resize and encode image
        processed_image = resize_image(image)
        img_base64 = encode_image_to_base64(processed_image)
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # OpenAI's vision model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analyze this image and respond in {language}. User question: {user_prompt}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"OpenAI Vision Error: {str(e)}"

def process_image_with_vision(image, user_prompt, language="English"):
    """Process image with vision model"""
    try:
        # Resize image for optimal processing
        processed_image = resize_image(image)
        
        # Convert to base64
        img_base64 = encode_image_to_base64(processed_image)
        
        # Create vision message
        vision_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"Analyze this image and respond in {language}. User question: {user_prompt}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                }
            ]
        )
        
        # Try with the primary vision model first
        try:
            response = vision_model.invoke([vision_message])
            return f"🤖 **Groq Vision**: {response.content}"
        except Exception as vision_error:
            # If Groq vision model fails, try Google Gemini as first backup
            if google_model:
                st.info("🔄 Groq vision unavailable, trying Google Gemini Vision...")
                google_result = process_image_with_google(image, user_prompt, language)
                if google_result and not google_result.startswith("Google Gemini Vision Error"):
                    return f"🔄 **Google Gemini Vision** (Backup): {google_result}"
            
            # If Google fails, try OpenAI as second backup
            if openai_client:
                st.info("🔄 Trying OpenAI Vision as secondary backup...")
                openai_result = process_image_with_openai(image, user_prompt, language)
                if openai_result and not openai_result.startswith("OpenAI Vision Error"):
                    return f"🔄 **OpenAI Vision** (Secondary Backup): {openai_result}"
            
            # If all vision models fail, provide helpful fallback
            if "decommissioned" in str(vision_error).lower() or "model" in str(vision_error).lower():
                return f"""❌ **All Vision Models Unavailable**

**Groq Error**: {str(vision_error)[:100]}...
**Google Gemini Status**: {"Available" if google_model else "Not configured"}
**OpenAI Status**: {"Available" if openai_client else "Not configured"}

🔄 **Alternative Options**:
1. **Describe the image** and I'll provide analysis based on your description
2. **Ask specific questions** about image analysis techniques  
3. **Try again later** - models may become available

💡 **What I can help with**:
- Image analysis concepts and techniques
- Photo composition advice
- Technical photography questions
- Image processing guidance

Please describe what you see in the image, and I'll provide detailed insights!"""
            else:
                raise vision_error
        
    except Exception as e:
        error_msg = str(e)
        return f"❌ Error processing image: {error_msg}"

def get_image_info(image):
    """Get basic image information"""
    width, height = image.size
    format_name = image.format or "Unknown"
    mode = image.mode
    
    # Calculate file size estimate
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    size_kb = len(img_bytes.getvalue()) / 1024
    
    return {
        "width": width,
        "height": height,
        "format": format_name,
        "mode": mode,
        "size_kb": round(size_kb, 2)
    }

def check_vision_model_availability():
    """Check if vision model is available"""
    try:
        # Try a simple test with the vision model
        test_message = HumanMessage(content="Test message")
        vision_model.invoke([test_message])
        return True, "Available"
    except Exception as e:
        error_msg = str(e).lower()
        if "decommissioned" in error_msg:
            return False, "Model Decommissioned"
        elif "rate limit" in error_msg:
            return False, "Rate Limited"
        elif "api key" in error_msg:
            return False, "API Key Issue"
        else:
            return False, f"Error: {str(e)[:50]}..."

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
    token_counter=text_model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

chain = (
    RunnablePassthrough.assign(
        messages=itemgetter("messages") | trimmer
    )
    | prompt
    | text_model
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
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []
if "image_analysis_mode" not in st.session_state:
    st.session_state.image_analysis_mode = False

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
    
    /* Image upload area */
    .image-upload-area {
        border: 2px dashed #667eea;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        margin: 1rem 0;
    }
    
    /* Image preview */
    .image-preview {
        border-radius: 0.75rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin: 1rem 0;
    }
    
    /* Image info card */
    .image-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        color: white;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    /* Vision mode indicator */
    .vision-mode {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 0.75rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
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
    
    # Vision model status
    if st.session_state.image_analysis_mode:
        vision_available, vision_status = check_vision_model_availability()
        if vision_available:
            st.success(f"🖼️ Groq Vision: {vision_status}")
        else:
            st.error(f"🖼️ Groq Vision: {vision_status}")
        
        # Show backup status
        if google_model:
            st.success("🔄 Google Gemini: Available")
        if openai_client:
            st.success("🔄 OpenAI Vision: Available")
    
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
    
    # Image Upload Section
    st.markdown("### 🖼️ Image Analysis")
    
    # Image analysis mode toggle
    image_mode = st.toggle("🔍 Vision Mode", value=st.session_state.image_analysis_mode)
    st.session_state.image_analysis_mode = image_mode
    
    if image_mode:
        st.success("🎯 Vision mode active - Upload images for analysis!")
        
        # Show current vision model info
        groq_status = "🟢 Available" if os.getenv("GROQ_API_KEY") else "🔴 No API Key"
        google_status = "🟢 Available" if google_model else "🔴 Not configured"
        openai_status = "🟢 Available" if openai_client else "🔴 Not configured"
        
        st.info(f"🤖 **Primary**: Groq Vision {groq_status}")
        st.info(f"🔄 **Backup 1**: Google Gemini {google_status}")
        st.info(f"🔄 **Backup 2**: OpenAI Vision {openai_status}")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "📸 Upload Image",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
            help="Upload an image to analyze with AI vision"
        )
        
        if uploaded_file is not None:
            try:
                # Load and display image
                image = Image.open(uploaded_file)
                
                # Display image preview
                st.image(image, caption=f"📁 {uploaded_file.name}", use_column_width=True)
                
                # Get image info
                img_info = get_image_info(image)
                
                # Display image information
                st.markdown(f"""
                <div class="image-info">
                    📊 <strong>Image Info:</strong><br>
                    📐 Size: {img_info['width']} × {img_info['height']} px<br>
                    📄 Format: {img_info['format']} ({img_info['mode']})<br>
                    💾 Size: {img_info['size_kb']} KB
                </div>
                """, unsafe_allow_html=True)
                
                # Store image in session state
                st.session_state.current_image = image
                st.session_state.current_image_name = uploaded_file.name
                
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
    else:
        st.info("💬 Text mode - Regular chat without image analysis")
        if "current_image" in st.session_state:
            del st.session_state.current_image
    
    st.divider()
    
    # Help section
    st.markdown("### 💡 Tips & Features")
    st.markdown("""
    ✨ **Features:**
    - Real-time date/time awareness
    - Weather data integration
    - Multi-language support
    - Chat history with context
    - 🖼️ **Triple Vision AI System**
    - 📸 **Multiple image format support**
    
    📝 **Commands:**
    - Type your question
    - Upload images in Vision Mode
    - Use 'Clear Chat' to reset
    - Use 'New Chat' for fresh session
    
    🖼️ **Vision Models:**
    - **Primary**: Groq Vision (Fast)
    - **Backup 1**: Google Gemini (Accurate)
    - **Backup 2**: OpenAI Vision (Reliable)
    - Automatic fallback between models
    
    🖼️ **Image Tips:**
    - Toggle Vision Mode to analyze images
    - Supports: PNG, JPG, JPEG, GIF, BMP, WEBP
    - Ask questions about uploaded images
    - Describe, analyze, or extract text from images
    """)

# Main Chat Area
st.markdown("""
<div class="title-main">
    <h1>🤖 AI ChatBot with Triple Vision & Real-Time Data</h1>
    <p>Intelligent assistant powered by Groq + Google Gemini + OpenAI - Fast, Accurate, Real-Time Aware + Advanced Image Analysis</p>
</div>
""", unsafe_allow_html=True)

# Display real-time status
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📅 Current Time", datetime.now().strftime("%H:%M:%S"))
with col2:
    st.metric("📆 Date", datetime.now().strftime("%Y-%m-%d"))
with col3:
    status = "✅ Online" if os.getenv("GROQ_API_KEY") else "❌ No API Key"
    st.metric("🔌 Status", status)
with col4:
    vision_status = "🖼️ Vision ON" if st.session_state.image_analysis_mode else "💬 Text Mode"
    st.metric("🤖 Mode", vision_status)

# Vision mode indicator
if st.session_state.image_analysis_mode:
    st.markdown("""
    <div class="vision-mode">
        🎯 VISION MODE ACTIVE - Upload images for AI analysis!
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Display chat history
with st.container():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
                # Display associated image if exists
                if "image" in msg:
                    st.image(msg["image"], caption="📸 Uploaded Image", width=300)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])

# Input area
st.divider()

col_input, col_send = st.columns([0.95, 0.05])

with col_input:
    if st.session_state.image_analysis_mode and "current_image" in st.session_state:
        user_input = st.chat_input(
            f"�️ Ask about the uploaded image: {st.session_state.current_image_name}",
            key="user_input",
            max_chars=2000
        )
    else:
        user_input = st.chat_input(
            "�💬 Type your message here (real-time data will be included)...",
            key="user_input",
            max_chars=2000
        )

# Process user input
if user_input:
    # Check if we're in vision mode with an image
    has_image = st.session_state.image_analysis_mode and "current_image" in st.session_state
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
        if has_image:
            st.image(st.session_state.current_image, caption=f"📸 {st.session_state.current_image_name}", width=300)
    
    # Store in session state
    user_message = {"role": "user", "content": user_input}
    if has_image:
        user_message["image"] = st.session_state.current_image
        user_message["image_name"] = st.session_state.current_image_name
    
    st.session_state.messages.append(user_message)
    
    # Get AI response
    with st.chat_message("assistant", avatar="🤖"):
        if has_image:
            # Use vision model for image analysis
            with st.spinner("🔍 Analyzing image with AI vision..."):
                try:
                    ai_response = process_image_with_vision(
                        st.session_state.current_image, 
                        user_input, 
                        language
                    )
                except Exception as e:
                    ai_response = f"❌ Vision Error: {str(e)}\n\n💡 Make sure GROQ_API_KEY is set and vision model is available"
        else:
            # Use regular text model with real-time context
            real_time_context = get_real_time_context()
            enhanced_input = f"{user_input}{real_time_context}"
            
            config = {"configurable": {"session_id": st.session_state.session_id}}
            
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
    
    # Clear current image after processing (optional - you can remove this if you want to keep the image)
    # if has_image:
    #     del st.session_state.current_image
    #     del st.session_state.current_image_name

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #999; font-size: 0.85rem;'>
    <p>🚀 Powered by Groq + Google Gemini + OpenAI + Streamlit | ⏰ Real-Time Aware | 🌍 Multi-Language | 🖼️ Triple Vision AI</p>
    <p>📡 Includes live date/time and weather data + Advanced multi-model image analysis for comprehensive responses</p>
    </div>
    """,
    unsafe_allow_html=True
)

# To run this file:
# streamlit run AIChatBot.py

