from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from graph.react import llm, tools

load_dotenv()

SYSTEM_MESSAGE = """
Sos un asistente agéntico útil que usa herramientas para responder preguntas. 
Tus respuestas deben incluir resúmenes del tema en cuestión.  
"""

def run_agent_reasoning(state:MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]])
    return {"messages": [response]}

tool_node = ToolNode(tools)