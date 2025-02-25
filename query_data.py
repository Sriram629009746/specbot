import argparse
#from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
import streamlit as st

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.

    st.set_page_config(page_title="Ask your question to nvme spec")
    st.header("Ask NVME Spec 📈")
    query_text = st.text_input("Ask your question here: ")
    submit_button = st.button("Submit Query")  # Add a submit button

    #parser = argparse.ArgumentParser()
    #parser.add_argument("query_text", type=str, help="The query text.")
    #args = parser.parse_args()
    #query_text = args.query_text
    if submit_button:
        if query_text is not None and query_text != "":
            with st.spinner(text="In progress..."):
                st.write(query_rag(query_text))



def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    #model = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")
    model = OllamaLLM(model="deepseek-r1:7b", base_url="http://localhost:11434")

    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
