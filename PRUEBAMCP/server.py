from mcp.server.fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import psutil
import os
from langchain_huggingface import HuggingFaceEmbeddings

mcp = FastMCP("TechSupportServer")
CHROMA_PATH = os.path.join(os.getcwd(), "chroma_db")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

@mcp.tool()
async def consultar_manuales(pregunta: str) -> str:
    """Busca en la documentación técnica para resolver dudas (RAG)."""
    docs = vectorstore.similarity_search(pregunta, k=3)
    return "\n---\n".join([d.page_content for d in docs])

@mcp.tool()
async def estado_servidor() -> str:
    """Herramienta NO RAG: Obtiene carga de CPU y RAM del sistema local."""
    return f"CPU: {psutil.cpu_percent()}% | RAM: {psutil.virtual_memory().percent}%"

@mcp.tool()
async def lista_documentos() -> list:
    """Herramienta RAG: Lista los nombres de archivos indexados en la DB."""
    data = vectorstore.get()
    return list(set(os.path.basename(m['source']) for m in data['metadatas']))

if __name__ == "__main__":
    mcp.run()