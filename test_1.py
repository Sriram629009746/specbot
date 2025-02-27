from langchain_experimental.agents import create_csv_agent
#from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import streamlit as st
#from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama


def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

#    OPENAI_API_KEY =

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")

    llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)
    #llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")
    #llm = ChatOllama(model="llama3.2", temperature=0)
    #llm = ChatOllama(model="gpt-3.5-turbo",temperature=0)
    if csv_file is not None:

        agent = create_csv_agent(llm, csv_file, verbose=True, allow_dangerous_code=True)#, agent_type=)

        user_question = st.text_input("Ask a question about your CSV: ")

        submit_button = st.button("Submit Query")  # Add a submit button

        if submit_button:
            if user_question is not None and user_question != "":
                with st.spinner(text="In progress..."):
                    st.write(agent.invoke(user_question))


if __name__ == "__main__":
    main()