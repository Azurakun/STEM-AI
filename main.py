import os
from dotenv import load_dotenv
import streamlit as st
from streamlit.components.v1 import html
from langchain_openai import ChatOpenAI
from crewai import Process, Crew
from agents import Agents
from tasks import Tasks
from tools import file_writer_tool

load_dotenv()

def main():

    st.set_page_config(
        page_title="STEM AI Specialist",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
            font-family: 'Arial', sans-serif;
        }
        h1 {
            color: #4b4b4b;
            text-align: center;
            padding: 20px 0;
        }
        .stTextArea {
            background-color: #ffffff;
        }
        p, li, h2, h3 {
            color: black;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("Lunar - The STEM Agent Specialist")


    question = st.text_area("What Topics do you want to know?", "")
    st.write(f"**Your Question:** {question}")

    st.markdown("Loading Answers...")

    openaigpt4 = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_NAME"),
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    stem_crew = Crew(
        agents=[Agents().stem_agent()],
        tasks=[Tasks(question).research_task()],
        process=Process.sequential,
        manager_llm=openaigpt4
    )
        
    results = stem_crew.kickoff()
    
    st.markdown("## Results obtained:")
    st.write(f"""
        **Answer:**
        {results}
    """)
if __name__ == "__main__":
    main()