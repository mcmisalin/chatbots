from langchain_google_vertexai import ChatVertexAI
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
import streamlit as st
import vertexai
from langchain.agents import tool, Tool, AgentType, initialize_agent


# Initialize Vertex AI
PROJECT_ID = "vaulted-zodiac-253111"
LOCATION = "us-central1"
STAGING_BUCKET = "gs:/immigration_pathways_agent_buckets"
DATA_STORE_ID = "immigration-pathways-python-based-tests_1720277320763"
LOCATION_ID = "global"

vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

@tool
def search_immigration_database(query: str) -> str:
    """Search for visa information."""
    from langchain_google_community import VertexAISearchRetriever

    retriever = VertexAISearchRetriever(
        project_id=PROJECT_ID,
        data_store_id=DATA_STORE_ID,
        location_id=LOCATION_ID,
        engine_data_type=2,
        max_documents=10,
    )

    result = str(retriever.invoke(query))

    return result

@st.cache_resource(show_spinner=False)
def LLM_init():
    template = """
    Your name is Immigration Pathways Chatbot. You are a visa and immigration expert in USA. 
    Your job is to establish all the visa types a client is eligible for.
    First, ask the user for necessary information to establish eligibility. 
    If you need additional data, use the tool "search_immigration_database" to retrieve the required information.
    Once all necessary data is gathered, provide a conclusive answer.
    {chat_history}
    Human: {human_input}
    Chatbot:"""

    promptllm = PromptTemplate(template=template, input_variables=["chat_history", "human_input"])
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    tools = [
        Tool(
            name="search_immigration_database",
            func=search_immigration_database,
            description="This tool provides information regarding visas and immigration to the USA from datastore."
        )
    ]
    
    try:
        llm = ChatVertexAI(model="gemini-1.5-flash-001")
    except Exception as e:
        print(f"Error initializing VertexAI: {e}")
        raise
    
    llm_chain = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )
    
    return llm_chain

st.set_page_config(page_title="💬 Immigration Pathways 💬")

st.title('💬 Immigration Pathways 💬')
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi, my name is Immigration Pathways Chatbot, how can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize LLM chain
    llm_chain = LLM_init()

    # Get response from the LLM chain
    response = llm_chain({"input": prompt})  # Ensure correct input key

    st.session_state.messages.append({"role": "assistant", "content": response["output"]})
    st.chat_message("assistant").write(response["output"])
