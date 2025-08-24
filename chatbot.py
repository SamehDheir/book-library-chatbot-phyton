import os
import streamlit as st
from streamlit_chat import message
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import warnings

warnings.filterwarnings("ignore")

st.title("Chatbot Book Library")
st.divider()

# ملف البيانات
data_file = "data.txt"

# تحميل النصوص
loader = TextLoader(data_file)
documents = loader.load()

# تقسيم النصوص إلى chunks صغيرة (لكل chunk ~500 حرف مع overlap 50)
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# فحص سريع للتأكد من تحميل المستندات
st.write(f"عدد المستندات بعد التقسيم: {len(docs)}")
for i, doc in enumerate(docs[:5]):  # عرض أول 5 مستندات فقط للتأكد
    st.write(f"Chunk {i+1}: {doc.page_content[:200]} ...")

# إنشاء embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# إنشاء vectorstore باستخدام FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

# إعداد retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# اختبار سريع للـ retriever
test_docs = retriever.get_relevant_documents("أنواع الكتب")
st.write(f"عدد المستندات المطابقة للاختبار: {len(test_docs)}")

# إنشاء الـ chain
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY")),
    retriever=retriever
)

# Session state
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ['Hello! I\'m your personal assistant built by Book Library.']

if 'past' not in st.session_state:
    st.session_state['past'] = ['Hello! How can I help you?']

# دالة المحادثة
def conversational_chat(prompt):
    result = chain({"question": prompt, "chat_history": st.session_state['history']})
    answer = result['answer']
    st.session_state['history'].append((prompt, answer))
    return answer

# واجهة Streamlit
request_container = st.container()
response_container = st.container()

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
