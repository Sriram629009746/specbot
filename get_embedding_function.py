from langchain_community.embeddings.ollama import OllamaEmbeddings
#from langchain_aws import BedrockEmbeddings
#from langchain_ollama.llms import OllamaLLM

from langchain_ollama import OllamaEmbeddings

#from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_function():
    #embeddings = HuggingFaceEmbeddings(
    #    model_name="sentence-transformers/all-mpnet-base-v2"
    #)

    embeddings = OllamaEmbeddings(
        #model="deepseek-r1:7b"
        #model="llama3.2"
        model="nomic-embed-text"
    )

    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
