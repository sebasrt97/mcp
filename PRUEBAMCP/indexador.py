
#### modificar, no cumple con la automatizacion por directorio ni con la comprobacion de chromaDB
#### Para evitar duplicados


### Usar una libreria como watchdog para vigilar la carpeta.
### antes de indexar verificar si el nombre del documento ya existe en la coleccion de chromaDb.


from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def crear_indice():
    # loader = DirectoryLoader(
    #     path_carpeta, 
    #     glob="./*.pdf", #ruta a la carpeta de los pdf + el filtro para los pedf
    #     loader_cls=PyPDFLoader
    # )
    
    loader = PyPDFLoader("/home/mike/Escritorio/agentes/ruedas_suecas.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    

    embeddings = OllamaEmbeddings(model="llama3.1:8b-instruct-q4_K_M",base_url="http://localhost:11434")

    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory="/home/mike/Escritorio/agentes/chroma_db" 
    )
    print("Base de datos guardada")

if __name__ == "__main__":
    crear_indice()