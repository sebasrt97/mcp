import os
from mcp.server.fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

mcp = FastMCP("Gestor_de_Notas")
NOTES_DIR = "PRUEBAMCP/documentos"
CHROMA_PATH = "/app/chroma_db"

if not os.path.exists(NOTES_DIR):
    os.makedirs(NOTES_DIR)


# Inicialización de RAG
embeddings = OllamaEmbeddings(model="llama3.1:8b-instruct-q4_K_M", base_url="http://host.docker.internal:11434")
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})    

@mcp.tool()
def listar_notas() -> str:
    """Lista archivos .txt disponibles."""
    try:
        files = [f for f in os.listdir(NOTES_DIR) if f.endswith('.txt')]
        return "\n".join(files) if files else "No hay notas."
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def leer_nota(nombre_archivo: str) -> str:
    """Muesta el contenido de una nota."""
    path = os.path.join(NOTES_DIR, nombre_archivo)
    try:
        if not os.path.exists(path):
            return "Error: El archivo no existe."
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error: {str(e)}"
    
@mcp.tool()
def listar_notas() -> str:
    """Herramienta no RAG: Lista archivos en la carpeta de notas."""
    files = os.listdir(NOTES_DIR)
    return "\n".join(files) if files else "No hay notas."

if __name__ == "__main__":
    mcp.run()
