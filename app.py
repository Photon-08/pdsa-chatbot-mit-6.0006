from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import streamlit as st



persist_direc ="chroma_db"
embeddings = SentenceTransformerEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1")

vectordb = Chroma(persist_directory=persist_direc, embedding_function=embeddings)


print("Db Fetched!")

repo_id = "google/gemma-7b"
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 1},
    huggingfacehub_api_token="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"
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)



st.header("Chatbot for IIT Madras BS Degree: Ask questions to 6.006 Intro to Algorithms by MIT")
st.caption("Developed by Indranil Bhattacharyya (21F1005840)")
st.text("Please note that this is an experimental app, if you find something inappropiate report it to the developer")

with st.sidebar:
    st.write("Chat history")
    l = len(st.session_state)

    if l>0:
        for i in st.session_state:
            
            with st.chat_message("human"):
                
                st.write(i)

            with st.chat_message('assitant'):
                st.write(st.session_state[i])
            st.divider() 

                



    
c = 0
prompt = st.chat_input("Say something")
if prompt:
    c += 1
    with st.spinner('Wait for it...'):
    
        result = qa_chain({"query":prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message('assistant'):
        st.write(result["result"])

    key_name = 'question' + str(c)
    st.session_state[prompt] = result["result"]
    

