import asyncio
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

# --- CONFIGURACIÓN DEL SERVIDOR MCP ---

server_params = StdioServerParameters(
    command="/home/bigdata/miniconda3/envs/mcp/bin/python",
    args=["/home/bigdata/mcp/PRUEBAMCP/server.py"], # Servidor mcp
    env=None
)
async def ainput(prompt: str) -> str:
    # Esto permite que el loop de asyncio siga corriendo mientras el usuario escribe
    return await asyncio.to_thread(input, prompt)

async def main():
    # Configurar el LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0,
        api_key="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", # O usa os.getenv("GROQ_API_KEY")
    )
    
    #  conexión MCP
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            

            # Esto importa automáticamente todas las tools que el servidor ofrece
            mcp_tools = await load_mcp_tools(session)
            

            # Configurar memoria del agente
            memory = MemorySaver()
            system_prompt = """Eres el 'Arquitecto de Sistemas Local'. 

                INSTRUCCIONES:
                - Para preguntas sobre archivos, utiliza 'consultar_manuales'. 
                - IMPORTANTE: Si tras usar una herramienta no encuentras la información, NO insistas. Informa al usuario de lo que has encontrado o de que no tienes acceso a ese detalle específico.
                - No entres en bucles infinitos de consulta."""
            
            app = create_react_agent(
                llm, 
                tools=mcp_tools, 
                checkpointer=memory,
                prompt=system_prompt
            )

            # Ejecutar consulta de prueba
            pregunta = "¿Qué herramientas tienes disponibles en el servidor MCP?"
            while True:
                # 1. Entrada de usuario asíncrona
                pregunta = await ainput("Pregunta>>> ")
                
                if pregunta.lower() in ["salir", "exit", "quit"]:
                    print("Chat terminado")
                    break
                config = {"configurable": {"thread_id": "mcp_user_1"}}
                
                print(f"--- Consulta: {pregunta} ---")
                
                inputs = {"messages": [("user", pregunta)]}
                
                # En entorno async usamos astream
                async for s in app.astream(inputs, config=config, stream_mode="values"):
                    message = s["messages"][-1]
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        print(f"   [MCP: Llamando a {message.tool_calls[0]['name']}...]")
                
                print(f"\nRESPUESTA FINAL:\n{message.content}")

if __name__ == "__main__":
    asyncio.run(main())