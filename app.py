from crewai import Crew, Process, Agent, Task
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os, json
load_dotenv()

os.environ['HF_TOKEN']="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"

def create_tool(docs):
    if docs:
        text_splitter_docs=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
        embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store=FAISS.from_documents(text_splitter_docs, embeddings)
        retriever=vector_store.as_retriever()
        retriever_tool=create_retriever_tool(retriever, "Analyzer and Table Writer", "Get related document information in the form of table")
        return retriever_tool

def create_agent(retriever_tool, llm):
    agent_analyzer=Agent(
    role='Analyze the provided information and return the output in the form of json',
    goal='give the output in the form of json with these columns Name, Experience, Number of Years and Skill Set from the provided information',
    backstory=(
       "Expert in understanding the information and provide the related information in the form of json" 
    ),
    verbose=True,
    memory=True,
    allow_delegation=True,
    tools=[retriever_tool],
    llm=llm
    )
    return agent_analyzer
def create_task(agent_analyzer):
    task_analyzer=Task(
    description='Analyze and return the information in the form of json',
    expected_output='The provided output in the form of json with these columns Name, Experience, Number of Years and Skill Set.',
    agent=agent_analyzer
    )
    return task_analyzer

if __name__=="__main__":

    st.set_page_config(page_title="Resume Summarizer", page_icon="🦜")
    st.title("🦜 Resume Summarizer")
    st.subheader('Summarize PDF Document')

    df = pd.DataFrame()

    with st.sidebar:
        groq_api_key=st.text_input("Groq Api key", value="", type="password")
    
    if groq_api_key:
        llm=ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192",streaming=True, temperature=0)
    else:
        llm=ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192",streaming=True)
        st.sidebar.error("Please Provide Groq Api Key")

    uploaded_files=st.file_uploader("Choose A PDf file", type="pdf", accept_multiple_files=True)

    documents=[]
    if uploaded_files:
        for upload_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,'wb') as file:
                file.write(upload_file.getvalue())
                file_name=upload_file.name
            try:
                loader=PyPDFLoader(temppdf)
                docs=loader.load()

                retriever_tool=create_tool(docs)
                agent_analyzer=create_agent(retriever_tool, llm)
                task_analyzer=create_task(agent_analyzer)
                crew=Crew(
                    agents=[agent_analyzer],
                    tasks=[task_analyzer],
                    process=Process.sequential
                )

                resp=crew.kickoff().raw
        
                # st.success(resp)

                start_index=resp.index("[")
                end_index=resp.index("]")
                json_array=resp[start_index:end_index+1]
                json_load=json.loads(json_array)
                # st.write(type(json_load))
                # st.write(json_load)
                data=pd.DataFrame(json_load)
                #st.dataframe(data)
                df=df._append(data, ignore_index = True)
            except Exception as e:
                st.write("The Error occured in : ", upload_file.name)
                st.write("The Error is : ", e)
    st.dataframe(df)
