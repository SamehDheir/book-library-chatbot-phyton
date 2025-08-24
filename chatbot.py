import os
import streamlit as st
from streamlit_chat import message
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import warnings

warnings.filterwarnings("ignore")

st.title("Chatbot Book Library")
st.divider()

data_file = "data.txt"

request_container = st.container()
response_container = st.container()

# Load documents
loader = TextLoader(data_file)
documents = loader.load()

# Create embeddings using OpenAI API key from Streamlit Secrets
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Create vectorstore index
index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])

# Create chain with k=3 to improve matching even with slight differences
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    ),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ['Hello! I\'m your personal assistant built by Book Library.']

if 'past' not in st.session_state:
    st.session_state['past'] = ['Hello! How can I help you?']

# Conversational chat function with strict source checking
def conversational_chat(prompt):
    result = chain({"question": prompt, "chat_history": st.session_state['history']})
    
    # Check if retriever returned any source documents
    if "source_documents" in result and result["source_documents"]:
        answer = result['answer']
    else:
        answer = "عذراً، لا أستطيع الإجابة عن هذا السؤال بناءً على محتوى مكتبة الكتب لدينا."
    
    st.session_state['history'].append((prompt, answer))
    return answer

# Streamlit UI
with request_container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("You: ", key="input")
        submit_button = st.form_submit_button("Send")

if submit_button and user_input:
    output = conversational_chat(user_input)
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)

if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(
                st.session_state['past'][i],
                is_user=True,
                key=str(i) + '_user',
                avatar_style="adventurer",
                seed=13
            )
            message(
                st.session_state['generated'][i],
                key=str(i),
                avatar_style="adventurer",
                seed=2
            )
