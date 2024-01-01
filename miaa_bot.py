
from langchain.document_loaders import PyPDFDirectoryLoader
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import pickle
import requests
import re

def picklr(file, ob):
    with open(file, 'wb') as f:
        pickle.dump(ob, f)

def unpicklr(file):
    with open(file, 'rb') as f:
        ob = pickle.load(f)
    return ob

LLM_MODEL = "llama2"
LLM_PROMPT = """Eres el asistente virtual del máster de Inteligencia Artificial Aplicada en la UC3M. Los estudiantes pueden hacerte preguntas sobre las asignaturas, y tú responderas con la información que te proporciono como contexto. Sé extremadamente breve. Sólo hablas español. Contexto: {context} Pregunta: {question} Respuesta: """

def llm_call(question, context):
    data = {
            "model": LLM_MODEL,
            "prompt": LLM_PROMPT.format(question = question, context = context),
            "stream": False
        }

    response = requests.post("http://localhost:11434/api/generate", json=data)
    print(response.json()["response"], "\n")
    return response


SENTENCE_TRANSFORMER = SentenceTransformer("intfloat/multilingual-e5-large")
TOP_SCORES = 1
CORPUS = unpicklr("./ingested_data/corpus.pkl")
VECS = unpicklr("./ingested_data/vecs.pkl")

def query(question):    
    target_vector = SENTENCE_TRANSFORMER.encode(question)
    cosine_similarities = util.pytorch_cos_sim(target_vector, VECS)[0]
    document_similarity_pairs = list(zip(CORPUS, cosine_similarities.tolist()))
    sorted_pairs = sorted(document_similarity_pairs, key=lambda x: x[1], reverse=True)
    similar_docs = sorted_pairs[:TOP_SCORES]

    for i, (context, score) in enumerate(similar_docs, start=1):
        print(f"Pregunta: {question}\nSimilarity Score: {score:.4f}\n{context}\n")
        response = llm_call(question, context)
        return response.json()["response"]

if __name__ == "__main__":

    st.title("UC3M MIAA Bot")
    if prompt := st.chat_input("¿Qué quieres saber?"):
        st.chat_message("user").markdown(prompt)
        response = query(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)