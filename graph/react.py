import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from datetime import datetime

load_dotenv()

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


tools = [TavilySearch(max_results=1), time_tool]

llm = ChatOllama(model="qwen3:8b",
                 api_key=os.getenv("OLLAMA_API_KEY"),
                 temperature=0).bind_tools(tools)
