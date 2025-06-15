import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted # Import the specific exception for rate limiting
import base64
import time

# --- Configuration ---
API_KEYS = [
    st.secrets.get("API_KEY_01"),
    st.secrets.get("API_KEY_02"),
    st.secrets.get("API_KEY_03"),
    st.secrets.get("API_KEY_04"),
    st.secrets.get("API_KEY_05")
]
# Filter out any keys that are not set (are None or empty strings)
API_KEYS = [key for key in API_KEYS if key]

MODEL_NAME = st.secrets["MODEL_NAME"]
BACKGROUND_IMAGE_PATH = "garage.jpeg"  # Create this image file

# --- Validate API Keys ---
if not API_KEYS:
    st.error("Please provide at least one GOOGLE_API_KEY in st.secrets (e.g., API_KEY_01).")
    st.stop()


# --- Agent and Tool Factory Function ---
def get_agent_executor_with_key(api_key):
    """
    Creates and returns an agent executor instance configured with the provided API key.
    This function is called to initialize or re-initialize the agent when a key rotation is needed.
    All tools are defined inside this function to ensure they are created with the latest,
    non-rate-limited LLM instance.
    """
    # --- Initialize Google Generative AI with the specific key ---
    os.environ["GOOGLE_API_KEY"] = api_key
    try:
        # Pass the client to the LLM to ensure it uses the correct configuration
        genai.configure(api_key=api_key)
        client = genai
    except Exception as e:
        st.warning(f"Error configuring Google Generative AI with a key: {e}")
        return None # Return None if a key is invalid

    # --- Tool Definitions (Defined here to use the correct LLM instance) ---
    @tool
    def greet_tool(_input: str = "") -> str:
        """
        Use this tool when the user offers a greeting, says hello, or when initiating the conversation.
        The '_input' argument for this tool can be empty as the greeting is standardized.
        Responds with a welcome message and initial prompt.
        Output in 3–5 bullet points.
        """
        res = """
        Welcome to Maverick’s IntelliTune Garage!\n
        • I am your AI service assistant.\n
        • How can I help with your vehicle today?\n
        • Type 'help' for available services or 'exit' to quit.
        """
        return res

    @tool
    def search_engine_problems(query: str) -> str:
        """
        Use this tool to analyze engine-related complaints from the user.
        Examples: 'car won't start', 'engine is making a strange noise', 'check engine light is on'.
        The 'query' argument should be the user's description of the engine problem.
        Responds with 3–5 bullet points on possible causes or checks; may ask one follow-up question if needed.
        """
        llm_tool = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1, client=client)
        prompt_text = f"""
    You are Maverick’s IntelliTune Garage AI.
    🔧 **Engine Complaint Analyzer**
    User says: \"{query}\"

    • If you need one concise follow-up question to better understand the problem, ask it.
    • Otherwise, respond with 3–5 concise bullet points on possible causes or checks.
    • End with: \"Please contact us to get this fixed or for more info.\"
    """
        return llm_tool.invoke([HumanMessage(content=prompt_text)]).content.strip()

    @tool
    def schedule_service(query: str) -> str:
        """
        Use this tool when the user wants to schedule a service or asks about recommended maintenance intervals.
        Examples: 'I need to book an oil change', 'When is my next service due?', 'What maintenance is needed at 50000 miles?'.
        The 'query' argument should be the user's request related to scheduling or maintenance.
        Responds with 3–5 bullet points on recommended maintenance; may ask one follow-up question if needed.
        """
        llm_tool = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1, client=client)
        prompt_text = f"""
    You are Maverick’s IntelliTune Garage AI.
    🛠️ **Scheduled Service Scheduler**
    User query: \"{query}\"

    • If you need one concise follow-up question (e.g., about vehicle make/model/year or mileage), ask it.
    • Otherwise, respond with 3–5 concise bullet points on recommended maintenance.
    • End with: \"Please contact us to get this fixed or for more info.\"
    """
        return llm_tool.invoke([HumanMessage(content=prompt_text)]).content.strip()

    @tool
    def assess_damage(query: str) -> str:
        """
        Use this tool when the user describes accident damage to their vehicle.
        Examples: 'I had a fender bender', 'My bumper is cracked', 'Someone hit my car'.
        The 'query' argument should be the user's description of the accident or damage.
        Responds with 3–5 bullet points assessing potential damage areas or next steps; may ask one follow-up question if needed.
        """
        llm_tool = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1, client=client)
        prompt_text = f"""
    You are Maverick’s IntelliTune Garage AI.
    🚧 **Accident Damage Assessor**
    User says: \"{query}\"

    • If you need one concise follow-up question to better understand the damage, ask it.
    • Otherwise, respond with 3–5 concise bullet points assessing potential damage or advice.
    • End with: \"Please contact us to get this fixed or for more info.\"
    """
        return llm_tool.invoke([HumanMessage(content=prompt_text)]).content.strip()

    @tool
    def routine_service(query: str) -> str:
        """
        Use this tool when the user asks about routine service checks or specific routine maintenance tasks.
        Examples: 'What should I check regularly?', 'Tell me about tire rotation', 'How often to check oil?'.
        The 'query' argument should be the user's question about routine service.
        Responds with 3–5 bullet points on routine checks; may ask one follow-up question if needed.
        """
        llm_tool = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1, client=client)
        prompt_text = f"""
    You are Maverick’s IntelliTune Garage AI.
    🔄 **Routine Service Coordinator**
    User asks: \"{query}\"

    • If you need one concise follow-up question, ask it.
    • Otherwise, respond with 3–5 concise bullet points on routine checks.
    • End with: \"Please contact us to get this fixed or for more info.\"
    """
        return llm_tool.invoke([HumanMessage(content=prompt_text)]).content.strip()

    @tool
    def contact_info(_input: str = "") -> str:
        """
        Use this tool when the user asks for contact details, address, phone number, or opening hours of the garage.
        The '_input' argument for this tool can be empty as the contact information is static.
        Returns contact details in bullet points.
        """
        address = """
        📍 Maverick’s IntelliTune Garage, Hesaraghatta Main Road, Bengaluru  \n
        🕙 10 AM – 6 PM (Weekdays)  \n
        📞 +91 98765 00000  \n
        🌐 www.intellitune.com  \n
        ✉️ intellitune@tuning.com  \n
        Please contact us for appointments or further information
        """
        return address

    tools = [greet_tool, search_engine_problems, schedule_service, assess_damage, routine_service, contact_info]

    system_prompt_text = """You are Maverick Agentic AI, a helpful and friendly AI service assistant for Maverick’s IntelliTune Garage.
    You have access to a variety of tools to help users with their vehicle inquiries.
    Based on the user's message, decide if one of your tools is appropriate to use.
    If you use a tool, its output will be your response.
    If the user just says hi or hello, or if it's the start of the conversation, use the greet_tool.
    If asked for help about services, list the types of queries you can handle based on your tools.
    If the user says 'bye', 'thank you', 'no more questions', or similar, respond with a polite closing statement.
    """
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm_agent = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, client=client)
    agent = create_tool_calling_agent(llm_agent, tools, agent_prompt)
    
    # Use existing memory if it's already in session state
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", k=20, return_messages=True)

    executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=False, handle_parsing_errors=True)
    return executor


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Agdasima');
    .custom-title-container { text-align: center !important; width: 100% !important; margin-top: 20px !important; }
    .custom-title-box { display: inline-block !important; background-color: rgba(0, 0, 0, 0.8) !important; padding: 10px 20px !important; border-radius: 5px !important; }
    .custom-title { font-family: 'Agdasima', sans-serif !important; font-size: 50px !important; color: cyan !important; margin: 0 !important; }
    </style>
    <div class="custom-title-container"><div class="custom-title-box"><p class="custom-title">Agentic AI for Maverick's IntelliTune Garage</p></div></div>
""", unsafe_allow_html=True)

def set_bg_from_local(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as image:
            encoded_string = base64.b64encode(image.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/{"png" if image_file.endswith(".png") else "jpg"};base64,{encoded_string});
                background-size: cover; background-repeat: no-repeat; background-attachment: fixed;
            }}
            .stChatMessage {{ background-color: rgba(255, 255, 255, 0.88) !important; border-radius: 10px; padding: 12px !important; margin-bottom: 10px; border: 1px solid #cccccc; }}
            .stChatMessage p, .stChatMessage li, .stChatMessage div[data-testid="stMarkdownContainer"] > div {{ color: #1e1e1e !important; font-size: 1rem !important; }}
            .stChatInputContainer {{ background-color: rgba(230, 230, 230, 0.90) !important; border-top: 1px solid #bbbbbb !important; }}
            div[data-testid="stChatInput"] textarea[aria-label="chat input"] {{ color: #1e1e1e !important; background-color: rgba(255, 255, 255, 0.95) !important; border-radius: 8px !important; border: 1px solid #aaaaaa !important; padding: 8px 12px !important; }}
            div[data-testid="stChatInput"] textarea[aria-label="chat input"]::placeholder {{ color: #555555 !important; opacity: 1 !important; }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f"Background image '{image_file}' not found. Using default background.")

set_bg_from_local(BACKGROUND_IMAGE_PATH)

# --- Session State Initialization ---
if "session_active" not in st.session_state:
    st.session_state.api_keys = API_KEYS
    st.session_state.current_api_key_index = 0
    st.session_state.agent_executor = None
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", k=20, return_messages=True)
    st.session_state.session_active = False

def initialize_chat():
    if not st.session_state.session_active:
        st.session_state.messages = []
        st.session_state.memory.clear()
        
        # Initial agent setup with the first key
        current_key = st.session_state.api_keys[st.session_state.current_api_key_index]
        st.session_state.agent_executor = get_agent_executor_with_key(current_key)
        
        if not st.session_state.agent_executor:
             st.error("Failed to initialize the agent with the first API key. Please check your keys.")
             st.stop()

        try:
            with st.spinner("AgenticAI is starting up..."):
                initial_greeting_response = st.session_state.agent_executor.invoke(
                    {"input": "User has just started the chat, greet them."})
            assistant_response = initial_greeting_response.get("output", "Sorry, I couldn't start up correctly.")
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        except Exception as e:
            st.error(f"Error during initial greeting from agent: {e}")
            st.session_state.messages.append(
                {"role": "assistant", "content": "I seem to be having trouble starting. Please try refreshing."})
        
        st.session_state.session_active = True
        st.rerun()

# --- Main Chat Logic ---
if not st.session_state.session_active:
    initialize_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help with your vehicle today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AgenticAI is thinking..."):
            response_content = ""
            if prompt.lower() in {"exit", "quit"}:
                response_content = "Goodbye! We look forward to helping you again."
                st.session_state.session_active = False
            elif prompt.lower() == "help":
                response_content = """
                I can help with:  \n
                • Analyzing engine complaints  \n
                • Scheduling services or asking about maintenance  \n
                • Assessing accident damage  \n
                • Answering routine service questions  \n
                • Providing our contact information  \n
                How can I assist you?
                """
            else:
                # --- CORE CIRCULAR KEY ROTATION AND RETRY LOGIC ---
                response = None
                start_index = st.session_state.current_api_key_index
                num_keys = len(st.session_state.api_keys)

                # This loop acts as our "circuit breaker". It will try each key once per user request.
                for i in range(num_keys):
                    # Calculate the next key index in a circular fashion
                    key_index_to_try = (start_index + i) % num_keys
                    
                    try:
                        # Re-initialize the agent with the key we are about to try
                        st.session_state.agent_executor = get_agent_executor_with_key(st.session_state.api_keys[key_index_to_try])
                        if not st.session_state.agent_executor:
                            st.warning(f"Skipping an invalid API key at index {key_index_to_try}.")
                            continue # Skip to the next key

                        # Attempt the call
                        response = st.session_state.agent_executor.invoke({"input": prompt})
                        
                        # If successful, update the session's current key index and break the loop
                        st.session_state.current_api_key_index = key_index_to_try
                        break
                    except ResourceExhausted:
                        st.warning(f"API rate limit hit on Key #{key_index_to_try + 1}. Rotating...")
                        time.sleep(1) # Small delay before trying the next key
                        # The loop will automatically continue to the next key
                        continue
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                        response_content = "I'm having some trouble processing that. Please try rephrasing."
                        break

                # After the loop, check if we got a response
                if response:
                    response_content = response.get("output", "Sorry, I didn't quite understand that.")
                # If the loop finished and response is still None, it means all keys failed
                elif not response_content:
                    st.error("All API keys are currently rate-limited. The system is under heavy load.")
                    response_content = "I'm currently experiencing very high traffic and can't process your request. Please try again in a minute."

            st.session_state.messages.append({"role": "assistant", "content": response_content})
            st.markdown(response_content)
            if not st.session_state.session_active:
                st.rerun()

# --- Session Reset Buttons ---
if not st.session_state.session_active:
    if st.button("Start New Session", key="main_reset_button"):
        st.session_state.current_api_key_index = 0 # Reset key index
        st.rerun()
elif st.button("Start New Session", key="manual_reset_button_active_session"):
    st.session_state.session_active = False
    st.session_state.current_api_key_index = 0 # Reset key index
    st.rerun()
