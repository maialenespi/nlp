from langchain.document_loaders import PyPDFDirectoryLoader
from sentence_transformers import SentenceTransformer, util
import re
import pickle


# Utilities para almacenar los datos procesados
def picklr(file, ob):
    with open(file, 'wb') as f:
        pickle.dump(ob, f)

def unpicklr(file):
    with open(file, 'rb') as f:
        ob = pickle.load(f)
    return ob

# Carga de datos de las fichas
loader = PyPDFDirectoryLoader("fichas/")
docs = loader.load()
my_dict = {}
for doc in docs:
    try:
        my_dict[doc.metadata["source"]] += doc.page_content
    except:
        my_dict[doc.metadata["source"]] = doc.page_content

# Carga de modelo transformer de generación de embeddings
SENTENCE_TRANSFORMER = SentenceTransformer("intfloat/multilingual-e5-large")

# Identificación de secciones para extraer chunks
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

    # Extracción nombre de la asignatura
    title = texto[: texto.find("Departamento")].upper()

    for sec, next_sec in zip(secciones, secciones[1:] + [None]):

        # Identificación start y end de sección
        start_index = texto.find(sec)
        end_index = texto.find(next_sec) if next_sec is not None else None
        
        if start_index != -1:

            # Extracción de contenido
            contenido = texto[start_index + len(sec):end_index]
            chunk = title + sec + contenido

            # Limpieza de chunk
            chunk = chunk.replace("\n", " ")
            chunk = re.sub(pag_regex, "", chunk)

            # Vectorización de chunk
            vec = SENTENCE_TRANSFORMER.encode(chunk)

            # Almacenado de texto y embeddings
            corpus.append(chunk)
            vecs.append(vec)

# Almacenado permanente de texto y embeddings
picklr("./ingested_data/corpus.pkl", corpus)
picklr("./ingested_data/vecs.pkl", vecs)
