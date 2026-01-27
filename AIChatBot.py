"""
AI ChatBot with Dual CNN & Web Research Integration
- Enhanced image output format with user-friendly responses
- Automatic cache clearing for each new query
- Internet-based responses in proper paragraph format with detailed information
- Complete standalone application with main execution function
"""


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
from PIL import Image
import io

# Try to import TensorFlow/Keras for ResNet50 and EfficientNet classification
try:
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50, EfficientNetB0
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess, decode_predictions as resnet_decode
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess, decode_predictions as efficientnet_decode
    from tensorflow.keras.preprocessing import image
    import numpy as np
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

load_dotenv()
warnings.filterwarnings("ignore")

# Web search functionality
def search_web_for_information(query, max_results=5):
    """Search the web for information about the identified object"""
    try:
        # Use DuckDuckGo API for web search
        search_url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        response = requests.get(search_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Extract relevant information
            search_results = []
            
            # Get abstract if available
            if data.get('Abstract'):
                search_results.append({
                    'type': 'abstract',
                    'content': data['Abstract'],
                    'source': data.get('AbstractSource', 'DuckDuckGo')
                })
            
            # Get related topics
            if data.get('RelatedTopics'):
                for topic in data['RelatedTopics'][:3]:  # Limit to 3 topics
                    if isinstance(topic, dict) and topic.get('Text'):
                        search_results.append({
                            'type': 'related',
                            'content': topic['Text'],
                            'source': 'DuckDuckGo'
                        })
            
            return search_results
        
        return []
        
    except Exception as e:
        print(f"Web search error: {e}")
        return []

def get_comprehensive_web_information(object_name):
    """Get comprehensive information from the internet"""
    try:
        # Search for general information
        general_query = f"{object_name} facts habitat diet behavior characteristics"
        search_results = search_web_for_information(general_query)
        
        # Search for specific aspects
        habitat_query = f"{object_name} habitat where found natural environment"
        habitat_results = search_web_for_information(habitat_query)
        
        diet_query = f"{object_name} diet food eating habits what eats"
        diet_results = search_web_for_information(diet_query)
        
        behavior_query = f"{object_name} behavior lifestyle interesting facts"
        behavior_results = search_web_for_information(behavior_query)
        
        # Combine all search results
        all_results = {
            'general': search_results,
            'habitat': habitat_results,
            'diet': diet_results,
            'behavior': behavior_results
        }
        
        return all_results
        
    except Exception as e:
        print(f"Information gathering error: {e}")
        return {}

# Groq text model for regular chat
text_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# Initialize ResNet50 and EfficientNet-B0 for image classification
resnet_model = None
efficientnet_model = None

if TENSORFLOW_AVAILABLE:
    try:
        # Load ResNet50 model with ImageNet weights
        resnet_model = ResNet50(weights='imagenet')
        print("✅ ResNet50 model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load ResNet50: {e}")
        resnet_model = None
    
    try:
        # Load EfficientNet-B0 model with ImageNet weights
        efficientnet_model = EfficientNetB0(weights='imagenet')
        print("✅ EfficientNet-B0 model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load EfficientNet-B0: {e}")
        efficientnet_model = None

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

def classify_image_with_resnet(image_pil):
    """Classify image using ResNet50 model"""
    if not resnet_model:
        return None
    
    try:
        # Resize image to 224x224 (ResNet50 input size)
        img_resized = image_pil.resize((224, 224))
        
        # Convert PIL image to array
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        
        # Expand dimensions to create batch
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess the image
        img_preprocessed = resnet_preprocess(img_array)
        
        # Make prediction
        predictions = resnet_model.predict(img_preprocessed, verbose=0)
        
        # Decode predictions to get top 5 classes
        decoded_predictions = resnet_decode(predictions, top=5)[0]
        
        # Format results
        results = []
        for i, (class_id, class_name, confidence) in enumerate(decoded_predictions):
            results.append({
                'rank': i + 1,
                'class': class_name.replace('_', ' ').title(),
                'confidence': float(confidence * 100)
            })
        
        return results
        
    except Exception as e:
        return f"ResNet50 Classification Error: {str(e)}"

def classify_image_with_efficientnet(image_pil):
    """Classify image using EfficientNet-B0 model"""
    if not efficientnet_model:
        return None
    
    try:
        # Resize image to 224x224 (EfficientNet-B0 input size)
        img_resized = image_pil.resize((224, 224))
        
        # Convert PIL image to array
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        
        # Expand dimensions to create batch
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess the image
        img_preprocessed = efficientnet_preprocess(img_array)
        
        # Make prediction
        predictions = efficientnet_model.predict(img_preprocessed, verbose=0)
        
        # Decode predictions to get top 5 classes
        decoded_predictions = efficientnet_decode(predictions, top=5)[0]
        
        # Format results
        results = []
        for i, (class_id, class_name, confidence) in enumerate(decoded_predictions):
            results.append({
                'rank': i + 1,
                'class': class_name.replace('_', ' ').title(),
                'confidence': float(confidence * 100)
            })
        
        return results
        
    except Exception as e:
        return f"EfficientNet-B0 Classification Error: {str(e)}"

def process_image_with_cnn_and_web_research(image, user_prompt, language="English"):
    """Process image using CNN models with web research for comprehensive responses"""
    try:
        # Get classifications from both CNN models
        resnet_results = None
        efficientnet_results = None
        
        st.info("🔍 Analyzing image with CNN models...")
        
        if resnet_model:
            resnet_results = classify_image_with_resnet(image)
        
        if efficientnet_model:
            efficientnet_results = classify_image_with_efficientnet(image)
        
        # Determine the primary identification from CNN models
        primary_object = None
        
        # Check EfficientNet results first (usually more accurate)
        if efficientnet_results and isinstance(efficientnet_results, list):
            primary_object = efficientnet_results[0]['class']
        elif resnet_results and isinstance(resnet_results, list):
            primary_object = resnet_results[0]['class']
        
        if primary_object:
            st.info(f"🎯 Identified: {primary_object}")
            st.info("🌐 Searching the web for comprehensive information...")
            
            # Get comprehensive information from the web
            web_info = get_comprehensive_web_information(primary_object)
            
            st.info("📝 Generating comprehensive response...")
            
            # Create a comprehensive prompt for the text model with web information
            synthesis_prompt = f"""
            Based on the CNN classification results and web research, create a comprehensive, engaging response about {primary_object}.
            
            CNN Classification Results:
            - Primary identification: {primary_object}
            - ResNet50 results: {resnet_results if resnet_results else 'Not available'}
            - EfficientNet-B0 results: {efficientnet_results if efficientnet_results else 'Not available'}
            
            Web Research Information:
            General: {web_info.get('general', [])}
            Habitat: {web_info.get('habitat', [])}
            Diet: {web_info.get('diet', [])}
            Behavior: {web_info.get('behavior', [])}
            
            User Question: {user_prompt}
            Language: {language}
            
            Please create a comprehensive response in {language} using this format:
            
            🔍 **What I See:**
            Start with a clear identification and description of what this {primary_object} is, including key characteristics that helped in identification.
            
            🌍 **Where It's From:**
            Describe the natural habitat, geographic distribution, and environmental preferences based on web research.
            
            🍽️ **Diet & Feeding:**
            Explain what it eats, how it feeds, and any interesting feeding behaviors or adaptations.
            
            ✨ **Notable Features:**
            Highlight distinctive physical characteristics, adaptations, and features that make it unique.
            
            🎭 **Behavior & Lifestyle:**
            Describe typical behaviors, social patterns, daily activities, and lifestyle characteristics.
            
            🤓 **Fun Facts:**
            Share interesting, educational, and surprising facts that make this subject fascinating.
            
            📊 **Classification Details:**
            Briefly mention that this was identified using advanced CNN models, but avoid technical details or confidence percentages.
            
            IMPORTANT: 
            - Write in engaging, flowing paragraphs (not bullet points)
            - Make it educational and accessible to general audiences
            - Use emojis and clear section headers
            - Avoid technical jargon and confidence percentages
            - Do NOT include any confidence levels, percentages, or technical accuracy metrics
            - Focus on making it interesting and informative
            - Include information from web research to make it comprehensive and current
            """
            
            try:
                response = text_model.invoke([HumanMessage(content=synthesis_prompt)])
                return response.content
            except Exception as e:
                # Fallback response if AI synthesis fails
                fallback_response = f"""🔍 **What I See:**
                Based on our CNN analysis, this appears to be a {primary_object}. This identification comes from analyzing the image using advanced deep learning models trained on millions of images.
                
                📊 **Classification Results:**
                Our dual CNN system provided the following analysis:
                """
                
                if resnet_results and isinstance(resnet_results, list):
                    fallback_response += f"\n🔍 **ResNet50 Top Predictions:**\n"
                    for i, result in enumerate(resnet_results[:3]):
                        fallback_response += f"{i+1}. {result['class']}\n"
                
                if efficientnet_results and isinstance(efficientnet_results, list):
                    fallback_response += f"\n⚡ **EfficientNet-B0 Top Predictions:**\n"
                    for i, result in enumerate(efficientnet_results[:3]):
                        fallback_response += f"{i+1}. {result['class']}\n"
                
                fallback_response += f"\n💡 **Note:** AI synthesis temporarily unavailable. Error: {str(e)[:100]}..."
                
                return fallback_response
        else:
            return """❌ **Unable to identify the main object in the image.**

🔍 **What happened:**
Our CNN models (ResNet50 and EfficientNet-B0) couldn't confidently identify the main subject in your image. This might happen when:

• The image quality is too low or blurry
• The object is not in our training dataset
• Unusual lighting or angle conditions
• The subject is partially obscured

🎯 **What you can try:**
1. **Upload a clearer image** with better lighting and focus
2. **Try a different angle** or closer shot of the subject
3. **Describe what you see** and I'll provide analysis based on your description
4. **Ask specific questions** about image analysis techniques

💡 **I can still help with:**
- General image analysis concepts and techniques
- Photography composition and lighting advice
- Technical questions about image processing
- Educational information about objects you describe

Feel free to describe what you see in the image, and I'll provide detailed insights based on your description!"""
        
    except Exception as e:
        return f"""❌ **Error during CNN analysis and web research**

🔧 **Technical Details:**
{str(e)}

🎯 **What you can try:**
1. **Check your internet connection** for web research functionality
2. **Ensure TensorFlow is properly installed** for CNN models
3. **Try uploading the image again**
4. **Contact support** if the issue persists

💡 **Alternative:** Describe the image content and I'll provide analysis based on your description."""
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

# Initialize session state with upload counter for cache clearing
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
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "image_processed" not in st.session_state:
    st.session_state.image_processed = False

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
    
    # CNN model status
    if st.session_state.image_analysis_mode:
        if resnet_model:
            st.success("� ResNet50 Classification: Loaded")
        else:
            st.error("� ResNet50: Not available")
        
        if efficientnet_model:
            st.success("⚡ EfficientNet-B0 Classification: Loaded")
        else:
            st.error("⚡ EfficientNet-B0: Not available")
        
        # Show TensorFlow status
        if TENSORFLOW_AVAILABLE:
            st.success("🧠 TensorFlow: Available")
        else:
            st.error("🧠 TensorFlow: Not installed")
    
    st.divider()
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_btn"):
            st.session_state.messages = []
            store[st.session_state.session_id] = ChatMessageHistory()
            # Clear image cache as well
            if "current_image" in st.session_state:
                del st.session_state.current_image
            if "current_image_name" in st.session_state:
                del st.session_state.current_image_name
            # Reset upload key to clear file uploader
            st.session_state.upload_key += 1
            st.session_state.image_processed = False
            st.rerun()
    
    with col2:
        if st.button("🔄 New Chat", use_container_width=True, key="new_btn"):
            st.session_state.session_id = f"chat_{datetime.now().timestamp()}"
            st.session_state.messages = []
            # Clear image cache as well
            if "current_image" in st.session_state:
                del st.session_state.current_image
            if "current_image_name" in st.session_state:
                del st.session_state.current_image_name
            # Reset upload key to clear file uploader
            st.session_state.upload_key += 1
            st.session_state.image_processed = False
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
    image_mode = st.toggle("🔍 CNN Mode", value=st.session_state.image_analysis_mode)
    st.session_state.image_analysis_mode = image_mode
    
    if image_mode:
        st.success("🎯 CNN mode active - Upload images for classification!")
        
        # Show current CNN model info
        resnet_status = "🟢 Loaded" if resnet_model else "🔴 Not available"
        efficientnet_status = "🟢 Loaded" if efficientnet_model else "🔴 Not available"
        tensorflow_status = "🟢 Available" if TENSORFLOW_AVAILABLE else "🔴 Not installed"
        
        st.info(f"� **CNN Model 1**: ResNet50 {resnet_status}")
        st.info(f"⚡ **CNN Model 2**: EfficientNet-B0 {efficientnet_status}")
        st.info(f"� **Framework**: TensorFlow {tensorflow_status}")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "� Upload Image",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
            help="Upload an image to analyze with CNN models",
            key=f"image_uploader_{st.session_state.upload_key}"
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
    - � **Dual CNN Classification System**
    - 📸 **Multiple image format support**
    
    📝 **Commands:**
    - Type your question
    - Upload images in CNN Mode
    - Use 'Clear Chat' to reset
    - Use 'New Chat' for fresh session
    
    � **CNN Models:**
    - **ResNet50**: Deep Learning classification (ImageNet trained)
    - **EfficientNet-B0**: Efficient & accurate classification
    - **TensorFlow**: Powered by Google's ML framework
    - Dual model comparison for better accuracy
    
    🖼️ **Image Analysis:**
    - Toggle CNN Mode to analyze images
    - Supports: PNG, JPG, JPEG, GIF, BMP, WEBP
    - Get dual CNN classification + detailed analysis
    - Compare ResNet50 vs EfficientNet predictions
    - Ask questions about uploaded images
    - AI-powered comprehensive object information
    """)

# Main Chat Area
st.markdown("""
<div class="title-main">
    <h1>🤖 AI ChatBot with Dual CNN & Web Research</h1>
    <p>Intelligent assistant powered by Groq + ResNet50 + EfficientNet-B0 + Web Research - Fast, Accurate, Real-Time Aware + Advanced CNN Image Classification with Internet-based Information</p>
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
    if st.session_state.image_analysis_mode:
        cnn_count = sum([resnet_model is not None, efficientnet_model is not None])
        if cnn_count == 2:
            vision_status = "🔍 Dual CNN ON"
        elif cnn_count == 1:
            vision_status = "� Single CNN ON"
        else:
            vision_status = "❌ No CNN Models"
    else:
        vision_status = "💬 Text Mode"
    st.metric("🤖 Mode", vision_status)

# CNN mode indicator
if st.session_state.image_analysis_mode:
    cnn_count = sum([resnet_model is not None, efficientnet_model is not None])
    if cnn_count == 2:
        st.markdown("""
        <div class="vision-mode">
            🔍 DUAL CNN MODE ACTIVE - Upload images for ResNet50 + EfficientNet-B0 classification!
        </div>
        """, unsafe_allow_html=True)
    elif cnn_count == 1:
        st.markdown("""
        <div class="vision-mode">
            🔍 SINGLE CNN MODE ACTIVE - Upload images for CNN classification!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="vision-mode">
            ❌ NO CNN MODELS AVAILABLE - Please install TensorFlow and restart the application!
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="vision-mode">
            🎯 TRIPLE VISION MODE ACTIVE - Upload images for AI analysis!
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
            f"🔍 Ask about the classified image: {st.session_state.current_image_name}",
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
            # Use CNN models with web research for comprehensive image analysis
            with st.spinner("🔍 Analyzing image with CNN models and web research..."):
                try:
                    ai_response = process_image_with_cnn_and_web_research(
                        st.session_state.current_image, 
                        user_input, 
                        language
                    )
                except Exception as e:
                    ai_response = f"❌ CNN Analysis Error: {str(e)}\n\n💡 Make sure TensorFlow is installed and internet connection is available for web research"
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
    
    # ENHANCED CACHE CLEARING: Clear current image cache after processing to prevent persistence
    if has_image:
        # Clear image from session state to prevent cache issues
        if "current_image" in st.session_state:
            del st.session_state.current_image
        if "current_image_name" in st.session_state:
            del st.session_state.current_image_name
        
        # Increment upload key to force file uploader reset
        st.session_state.upload_key += 1
        st.session_state.image_processed = True
        
        # Force a rerun to clear the UI state and prevent image persistence
        st.rerun()

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #999; font-size: 0.85rem;'>
    <p>🚀 Powered by Groq + ResNet50 + EfficientNet-B0 + TensorFlow + Web Research + Streamlit | ⏰ Real-Time Aware | 🌍 Multi-Language | 🔍 Dual CNN AI</p>
    <p>📡 Includes live date/time and weather data + Advanced CNN image classification + Web research integration + AI-powered comprehensive analysis</p>
    </div>
    """,
    unsafe_allow_html=True
)



# To run this application:
# streamlit run app_ui.py
