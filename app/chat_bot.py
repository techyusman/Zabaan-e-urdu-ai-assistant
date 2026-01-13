import streamlit as st
# --- UPDATED IMPORTS FOR NEW LANGCHAIN VERSIONS ---
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
# --------------------------------------------------
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
from dotenv import load_dotenv
import os
import re

# =========================
# 1. CONFIGURATION
# =========================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="Zabaan-e-Urdu",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
)

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found. Please set it in .env")
    st.stop()

# =========================
# 2. URDU-ELEGANT THEME (CSS)
# =========================
st.markdown("""
<style>
    /* IMPORTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Nastaliq+Urdu:wght@400;500;600;700&display=swap');

    /* --- ANIMATIONS --- */
    @keyframes messageSlideIn {
        0% {
            opacity: 0;
            transform: translateY(15px) scale(0.98);
        }
        100% {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    @keyframes gentlePulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    @keyframes typingDots {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-5px); }
    }
    
    @keyframes inputFocusGlow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(0, 0, 0, 0.1); }
        50% { box-shadow: 0 0 0 4px rgba(0, 0, 0, 0.1); }
    }
    
    @keyframes urduCalligraphy {
        0% { transform: scale(0.9) rotate(-2deg); opacity: 0; }
        100% { transform: scale(1) rotate(0); opacity: 1; }
    }

    /* --- GLOBAL SETTINGS --- */
    .stApp {
        background: #FFFFFF;
        color: #000000;
        font-family: 'Inter', sans-serif;
    }

    /* Hide Default Elements */
    #MainMenu, header, footer {visibility: hidden;}

    /* --- CHAT MESSAGES --- */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
        animation: messageSlideIn 0.4s ease-out forwards;
    }

    /* USER MESSAGE - Right Side */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        flex-direction: row-reverse;
    }
    
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) .stChatMessageContent {
        background: #000000;
        color: #FFFFFF;
        border-radius: 20px 20px 6px 20px;
        padding: 14px 18px;
        margin: 6px 0;
        max-width: 75%;
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* ASSISTANT MESSAGE - Left Side */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) .stChatMessageContent {
        background: #F8F9FA;
        color: #000000;
        border-radius: 20px 20px 20px 6px;
        padding: 14px 18px;
        margin: 6px 0;
        max-width: 75%;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.05);
    }

    /* TYPING INDICATOR */
    .typing-dots {
        display: inline-flex;
        gap: 3px;
        margin-left: 8px;
    }
    
    .typing-dot {
        width: 4px;
        height: 4px;
        background: #000000;
        border-radius: 50%;
        animation: typingDots 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }

    /* URDU TEXT STYLING */
    .urdu-text {
        font-family: 'Noto Nastaliq Urdu', serif;
        direction: rtl;
        line-height: 2.2;
        font-size: 1.12rem;
        letter-spacing: 0.3px;
    }
    
    .urdu-title {
        font-family: 'Noto Nastaliq Urdu', serif;
        direction: rtl;
        font-size: 2.5rem;
        font-weight: 600;
        line-height: 1.3;
        animation: urduCalligraphy 1s ease-out forwards;
    }

    /* --- MODERN INPUT CONTAINER --- */
    .stChatInputContainer {
        position: relative;
        background: #FFFFFF;
        border-radius: 24px;
        border: 1.5px solid #E5E5E5;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        margin-top: 20px;
        padding-right: 100px !important;
    }
    
    .stChatInputContainer:focus-within {
        border-color: #000000;
        box-shadow: 0 12px 40px rgba(0,0,0,0.12);
        animation: inputFocusGlow 2s infinite;
    }

    [data-testid="stChatInput"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding-right: 100px !important;
    }
    
    [data-testid="stChatInput"] textarea {
        color: #000000 !important;
        font-weight: 400;
        padding: 16px 20px;
        font-size: 0.95rem;
        line-height: 1.5;
        background: transparent !important;
    }
    
    [data-testid="stChatInput"] textarea::placeholder {
        color: #888888;
        font-weight: 400;
    }

    /* --- INTEGRATED MICROPHONE BUTTON --- */
    .mic-button-integrated {
        position: absolute !important;
        right: 50px !important;
        bottom: 8px !important;
        z-index: 1002 !important;
        background: transparent !important;
        border: none !important;
        color: #000000 !important;
        font-size: 1.1rem !important;
        padding: 6px !important;
        width: 32px !important;
        height: 32px !important;
        transition: all 0.3s ease !important;
        opacity: 0.7;
        border-radius: 8px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .mic-button-integrated:hover {
        opacity: 1;
        background: rgba(0,0,0,0.05) !important;
        transform: scale(1.05);
    }

    /* Send Button Styling */
    .stChatInputSubmitButton {
        position: absolute !important;
        right: 10px !important;
        bottom: 8px !important;
        z-index: 1001 !important;
        color: #000000 !important;
        background: transparent !important;
        border: none !important;
        padding: 6px !important;
        width: 32px !important;
        height: 32px !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    .stChatInputSubmitButton:hover {
        background: rgba(0,0,0,0.05) !important;
        transform: scale(1.05);
    }

    /* Audio Player Styling */
    .stAudio {
        margin-top: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        background: #000000;
        color: #FFFFFF;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .status-dot {
        width: 6px;
        height: 6px;
        background: #00D26A;
        border-radius: 50%;
        animation: gentlePulse 2s infinite;
    }

    /* Welcome Message */
    .welcome-card {
        background: #F8F9FA;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        border: 1px solid rgba(0,0,0,0.05);
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    }
    
    .urdu-subtitle {
        font-family: 'Noto Nastaliq Urdu', serif;
        direction: rtl;
        font-size: 1.1rem;
        color: #666666;
        line-height: 1.8;
    }

</style>
""", unsafe_allow_html=True)

# =========================
# 3. LOGIC (UNCHANGED)
# =========================
def clean_text_for_speech(text: str) -> str:
    cleaned = re.sub(r"[*.,!?:؟،۔(){}[\\]\"'<>#@%^&+=_/\\|~-]", "", text)
    return cleaned.strip()

def is_app_related_query(query: str) -> bool:
    keywords = ["who created you", "who developed you", "who made you", "اپ کو کس نے ڈیزائن کیا ہے", "آپ کو کس نے بنایا"]
    return any(k in query.lower() for k in keywords)

# =========================
# 4. URDU-ELEGANT HEADER
# =========================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 30px 0 20px 0;">
        <div class="urdu-title">زبانِ اردو</div>
        <div style="display: flex; align-items: center; justify-content: center; gap: 8px; margin-top: 12px;">
            <span class="status-badge">
                <span class="status-dot"></span>
                AI Language Assistant
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Welcome message for empty chat
if not st.session_state.get("langchain_messages", []):
    st.markdown("""
    <div class="welcome-card">
        <div class="urdu-subtitle">خوش آمدید! میں آپ کا اردو معاون ہوں</div>
        <p style="font-family:'Inter'; color:#666666; margin:15px 0 0 0; font-size: 0.95rem;">
            Your intelligent Urdu language companion for meaningful conversations
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# 5. CHAT ENGINE (UNCHANGED)
# =========================
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("You are a helpful AI assistant named 'Zabaan-e-Urdu'. Respond in pure, formal Urdu with elegant language."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)
msgs = StreamlitChatMessageHistory(key="langchain_messages")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
chain = prompt | model | StrOutputParser()
chain_with_history = RunnableWithMessageHistory(chain, lambda session_id: msgs, input_messages_key="question", history_messages_key="chat_history")

# Render History
for msg in msgs.messages:
    role = "human" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(f'<div class="urdu-text">{msg.content}</div>', unsafe_allow_html=True)

# =========================
# 6. INPUT UI & PROCESSING
# =========================

# Custom input container with integrated microphone
st.markdown("""
<div style="position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); width: 80%; max-width: 800px; z-index: 1000;">
    <div class="stChatInputContainer">
""", unsafe_allow_html=True)

# Text Input
user_text = st.chat_input("اپنا پیغام یہاں لکھیں...")

# Integrated microphone button
col1, col2 = st.columns([6, 1])
with col2:
    st.markdown("""
    <div style="position: absolute; right: 50px; bottom: 8px; z-index: 1002;">
    """, unsafe_allow_html=True)
    voice_text = speech_to_text(
        language="ur",
        start_prompt="🎤",
        stop_prompt="⏹️",
        use_container_width=False,
        just_once=True,
        key="STT"
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# Add some padding at the bottom for the fixed input
st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

# Processing Logic
final_input = user_text if user_text else voice_text

if final_input:
    with st.chat_message("human"):
        st.markdown(f'<div class="urdu-text">{final_input}</div>', unsafe_allow_html=True)

    if is_app_related_query(final_input):
        creator_response = "Mujhe M.Usman ne banaya hai"
        msgs.add_user_message(final_input)
        msgs.add_ai_message(creator_response)
        with st.chat_message("assistant"):
            st.markdown(f'<div class="urdu-text">{creator_response}</div>', unsafe_allow_html=True)
        
        clean_response = clean_text_for_speech(creator_response)
        tts = gTTS(text=clean_response, lang="ur")
        tts.save("output.mp3")
        st.audio("output.mp3", autoplay=False)

    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner(""):
                config = {"configurable": {"session_id": "any"}}
                response_stream = chain_with_history.stream({"question": final_input}, config)
                for chunk in response_stream:
                    full_response += chunk or ""
                    message_placeholder.markdown(
                        f'<div class="urdu-text">{full_response}<span class="typing-dots"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></span></div>', 
                        unsafe_allow_html=True
                    )
            message_placeholder.markdown(f'<div class="urdu-text">{full_response}</div>', unsafe_allow_html=True)

        try:
            clean_ai_response = clean_text_for_speech(full_response)
            if clean_ai_response:
                tts = gTTS(text=clean_ai_response, lang="ur")
                tts.save("output.mp3")
                st.audio("output.mp3", autoplay=False)
        except:
            pass