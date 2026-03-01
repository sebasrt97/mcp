
#### modificar, no cumple con la automatizacion por directorio ni con la comprobacion de chromaDB
#### Para evitar duplicados


### Usar una libreria como watchdog para vigilar la carpeta.
### antes de indexar verificar si el nombre del documento ya existe en la coleccion de chromaDb.

import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PATH = "/app/chroma_db"
DOCS_PATH = "PRUEBAMCP/documentos"

embeddings = OllamaEmbeddings(model="llama3.1:8b-instruct-q4_K_M", base_url="http://host.docker.internal:11434")
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

class NewDocHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".pdf"):
            fname = os.path.basename(event.src_path)
            # Requisito: Comprobar si ya ha sido procesado

            res = vectorstore.get(where={"source": event.src_path})
            if res and res['ids']:
                print(f"Archivo {fname} ya existe en ChromaDB. Saltando...")
                return
            
            print(f"Procesando nuevo documento: {fname}")
            loader = PyPDFLoader(event.src_path)
            splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
            vectorstore.add_documents(splits)
            print(f"Documento {fname} indexado con éxito.")

if __name__ == "__main__":
    observer = Observer()
    observer.schedule(NewDocHandler(), path=DOCS_PATH, recursive=False)
    observer.start()