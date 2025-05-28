## 🚗 Maverick's IntelliTune Garage AI
Welcome to Maverick's IntelliTune Garage AI, a Streamlit-based conversational assistant powered by Google Gemini and LangChain tools. This smart garage assistant helps users analyze car problems, schedule services, assess damage, and provide essential contact information—powered by the intelligence of Agentic AI.



🔧 Project Overview
This application serves as an AI-powered customer service representative for an automobile garage. Built with Streamlit, LangChain, and Gemini 2.5, it responds conversationally to vehicle-related queries and intelligently routes the requests to specialized tools.

✨ Features
👋 Friendly Greetings & Session Management

🚗 Engine Issue Analyzer: Diagnoses common engine problems based on user input.

🛠️ Scheduled Service Assistant: Offers maintenance advice or schedules based on mileage.

🌀 Routine Service FAQs: Educates users on regular checks like oil levels, tire rotation, etc.

💥 Damage Assessment: Provides suggestions for handling accident-related damage.

📞 Contact Info: Shares garage location, hours, and contact details.

🎨 Custom UI Styling: Personalized branding with background image and styled components.

💬 Persistent Chat Memory: Built-in memory to keep track of ongoing sessions.

🤖 What is Agentic AI?
Agentic AI refers to autonomous systems that can use tools, make decisions, and execute tasks intelligently on behalf of users. Rather than simply responding to prompts, agentic systems:

Understand user intent.

Choose from available tools.

Follow multi-step reasoning to solve problems.

Maintain context over time (e.g., conversations).

✅ How It Applies Here
Maverick’s IntelliTune Garage AI is a practical application of agentic AI:

It decides whether to greet, analyze an engine problem, or book a service.

It invokes the correct LangChain tool based on the query.

It maintains a chat memory, allowing it to understand the context and history.

It acts as an intelligent service agent, reducing manual intervention for customer support.

🧠 Tech Stack
Technology	Description
🧠 LangChain	Framework for building LLM-powered applications
🧠 Gemini 2.5 (Google Generative AI)	LLM used for generating contextual responses
🖼️ Streamlit	UI framework for creating data and AI apps
🧰 LangChain Tooling	Custom tools for each service functionality
🗂️ ConversationBufferMemory	Tracks chat history over sessions
🎨 Custom CSS	For UI styling and garage-themed appearance

🚀 Getting Started
🛠️ Prerequisites
Python 3.8+

A Google Gemini API Key

📦 Installation
bash
Copy
Edit
git clone https://github.com/your-username/intellitune-garage-ai.git
cd intellitune-garage-ai
pip install -r requirements.txt
🔑 Setup
Set your Google API key in the script or use an .env file:

bash
Copy
Edit
export GOOGLE_API_KEY=your_actual_key
Or replace the placeholder directly in the code.

▶️ Run the App
bash
Copy
Edit
streamlit run app.py
🖼️ UI Customization
Add a custom background image by placing your desired image in the appropriate location and updating the path:

python
Copy
Edit
BACKGROUND_IMAGE_PATH = r"path\to\your\garage.jpg"
📂 Folder Structure
text
Copy
Edit
├── app.py
├── requirements.txt
└── README.md
🧩 Future Enhancements
🔐 User authentication for booking services

📆 Calendar integration for appointment slots

📷 Image upload for damage detection

🔄 Integration with CRM or service ticketing systems
