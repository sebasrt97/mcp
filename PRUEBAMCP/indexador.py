import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# CONFIGURACIÓN LOCAL
CHROMA_PATH = os.path.join(os.getcwd(), "chroma_db")
DOCS_PATH = os.path.join(os.getcwd(), "documentos")
os.makedirs(DOCS_PATH, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

class NewDocHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".pdf"):
            self.procesar_archivo(event.src_path)

    def procesar_archivo(self, file_path):
        fname = os.path.basename(file_path)
        
        time.sleep(1)
        # REQUISITO: Comprobar si ya existe en ChromaDB para evitar duplicados
        res = vectorstore.get(where={"source": file_path})
        if res and res['ids']:
            print(f"[-] {fname} ya está indexado. Saltando...")
            return

        print(f"[+] Indexando nuevo documento: {fname}")
        loader = PyPDFLoader(file_path)
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
        
        # Añadimos la ruta al metadato 'source' para la comprobación
        for s in splits: s.metadata["source"] = file_path
        
        vectorstore.add_documents(splits)
        print(f"[OK] {fname} procesado con éxito.")

if __name__ == "__main__":
    handler = NewDocHandler()
    # Escaneo inicial de archivos existentes
    for f in os.listdir(DOCS_PATH):
        if f.endswith(".pdf"): handler.procesar_archivo(os.path.join(DOCS_PATH, f))
    
    observer = Observer()
    observer.schedule(handler, path=DOCS_PATH, recursive=False)
    observer.start()
    print(f"[*] Vigilando carpeta local: {DOCS_PATH}")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()