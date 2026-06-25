import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

load_dotenv()

_tavily = TavilySearch(max_results=3)

@tool
def time_tool() -> str:
    """
    Retorn información detallada sobre el día de hoy y la hora actual
    returns: Fecha, Hora y día de la semana
    """
    today_day = datetime.today()
    day_name = today_day.strftime("%A")
    date_str = today_day.strftime("%d de %B de %Y")
    time_str = today_day.strftime("%H:%M:%S")
    return f"Día {day_name}, {date_str} a las {time_str}"

@tool
def web_search(query: str) -> str:
    """
    Busca información actual en internet. Usala para datos que cambien con el tiempo.
    Usala para datos como el clima, noticias, precios, eventos recientes, entre otros.
    Para consultas de clima, formulá la query incluyendo la palabra "actual"
    o "ahora" (ej: "clima actual Buenos Aires") en lugar de "pronóstico",
    para obtener condiciones del momento y no un forecast extendido.
    
    Args:
        query: Consulta de búsqueda en lenguaje natural, sin fechas ni rangos
    """
    result = _tavily.invoke({"query": query})
    return str(result)
    
def get_llm():
    """
    Si OLLAMA_HOST está seteado (o está en modo local), usar Ollama.
    Sino, se usa Groq como fallback gratuito (Codespaces / Streamlit Cloud).
    """
    if os.getenv("USE_LOCAL_LLM") == "true":
        return ChatOllama(
            model="qwen3:8b",
            api_key=os.getenv("OLLAMA_API_KEY"),
            temperature=0
        )
    else:
        return ChatGroq(
            model="qwen3:8b",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )
        

tools = [web_search, time_tool]

llm = get_llm().bind_tools(tools)