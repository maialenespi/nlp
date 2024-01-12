
from langchain.document_loaders import PyPDFDirectoryLoader
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import pickle
import requests
import re

# Utilities para almacenar los datos procesados
def picklr(file, ob):
    with open(file, 'wb') as f:
        pickle.dump(ob, f)

def unpicklr(file):
    with open(file, 'rb') as f:
        ob = pickle.load(f)
    return ob

# Selección de modelo y prompt
LLM_MODEL = "llama2"
LLM_PROMPT = """Eres el asistente virtual del máster de Inteligencia Artificial Aplicada en la UC3M. Los estudiantes pueden hacerte preguntas sobre las asignaturas, y tú responderas con la información que te proporciono como contexto. Sé extremadamente breve. Sólo hablas español. Contexto: {context} Pregunta: {question} Respuesta: """

# Llamada a LLM
def llm_call(question, context):
    data = {
            "model": LLM_MODEL,
            "prompt": LLM_PROMPT.format(question = question, context = context),
            "stream": False
        }

    # POST a modelo corriendo en local (>> ollama run llama2)
    response = requests.post("http://localhost:11434/api/generate", json=data)
    print(response.json()["response"], "\n")
    return response

# Carga de modelo transformer de generación de embeddings (para pregunta)
SENTENCE_TRANSFORMER = SentenceTransformer("intfloat/multilingual-e5-large")

# Chunks de contexto 
TOP_SCORES = 1 

# Carga de textos de contexto
CORPUS = unpicklr("./ingested_data/corpus.pkl")

# Carga de vectores de contexto
VECS = unpicklr("./ingested_data/vecs.pkl")

# Búsqueda de vectores más cercanos
def search_closest_vecs(question):
    vec = SENTENCE_TRANSFORMER.encode(question) # Vectorización de pregunta
    cos_sim = util.pytorch_cos_sim(vec, VECS)[0] # Similaridad de coseno con los vectores guardados
    doc_sim = list(zip(CORPUS, cos_sim.tolist())) # Creación de parejas texto - similaridad
    sort_p = sorted(doc_sim, key=lambda x: x[1], reverse=True) # Ordenación parejas texto - similaridad
    similar_docs = sort_p[:TOP_SCORES] # Extracción textos más similares
    return similar_docs

# Llamada completa 
def query(question):    
    similar_docs = search_closest_vecs(question) # Búsqueda textos más similares (contexto)
    for i, (context, _) in enumerate(similar_docs, start=1):
        print(f"Pregunta: {question}\n{context}\n") 
        response = llm_call(question, context) # Llamada a LLM (contexto + pregunta)
        return response.json()["response"]



if __name__ == "__main__":

    # Front-end Streamlit
    st.title("UC3M MIAA Bot")
    if prompt := st.chat_input("¿Qué quieres saber?"):
        st.chat_message("user").markdown(prompt)
        response = query(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)