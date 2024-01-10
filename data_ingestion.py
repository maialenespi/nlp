from langchain.document_loaders import PyPDFDirectoryLoader
from sentence_transformers import SentenceTransformer, util
import re
import pickle


def picklr(file, ob):
    with open(file, 'wb') as f:
        pickle.dump(ob, f)

def unpicklr(file):
    with open(file, 'rb') as f:
        ob = pickle.load(f)
    return ob


loader = PyPDFDirectoryLoader("fichas/")


docs = loader.load()


my_dict = {}
for doc in docs:
    try:
        my_dict[doc.metadata["source"]] += doc.page_content
    except:
        my_dict[doc.metadata["source"]] = doc.page_content


SENTENCE_TRANSFORMER = SentenceTransformer("intfloat/multilingual-e5-large")


secciones = ["REQUISITOS (ASIGNATURAS O MATERIAS CUYO CONOCIMIENTO SE PRESUPONE)",
             "OBJETIVOS",
             "DESCRIPCIÓN DE CONTENIDOS: PROGRAMA",
             "ACTIVIDADES FORMATIVAS, METODOLOGÍA A UTILIZAR Y RÉGIMEN DE TUTORÍAS",
             "SISTEMA DE EVALUACIÓN",
             "BIBLIOGRAFÍA BÁSICA"]

pag_regex = re.compile(r'Página \d+ de \d+')
corpus = []
vecs = []

for _, texto in my_dict.items():
    title = texto[: texto.find("Departamento")].upper()

    for sec, next_sec in zip(secciones, secciones[1:] + [None]):
        start_index = texto.find(sec)
        end_index = texto.find(next_sec) if next_sec is not None else None
        
        if start_index != -1:
            contenido = texto[start_index + len(sec):end_index]
            chunk = title + sec + contenido
            chunk = chunk.replace("\n", " ")
            chunk = re.sub(pag_regex, "", chunk)
            vec = SENTENCE_TRANSFORMER.encode(chunk)
            corpus.append(chunk)
            vecs.append(vec)

picklr("./ingested_data/corpus.pkl", corpus)
picklr("./ingested_data/vecs.pkl", vecs)
