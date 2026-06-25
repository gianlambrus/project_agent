import re
import uuid
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessagesState, StateGraph, END
from graph.nodes import run_agent_reasoning, run_agent_reflection, run_format_output, run_sumarize, safe_tool_node


AGENT_REASON="agent_reason"
ACT="act"
REFLECT="reflect"
FORMATTER="formatter"
SUMARIZER="sumarizer"
LAST=-1

def should_continue(state:MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

def after_reflect(state: MessagesState) -> str:
    last = state["messages"][-1]
    content = last.content if hasattr(last, "content") else str(last)
    clean_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if "RESPUESTA_VALIDA" in clean_content:
        if len(state["messages"]) > 6:
            return SUMARIZER
        return FORMATTER
    return AGENT_REASON

def get_last_content(messages:list) -> str:
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and msg.content.strip():
            return msg.content.strip()
    return "No se pudo generar una respuesta precisa."

def clean_output(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def build_graph():
    graph_flow = StateGraph(MessagesState)
    graph_flow.add_node(AGENT_REASON, run_agent_reasoning)
    graph_flow.set_entry_point(AGENT_REASON)
    graph_flow.add_node(FORMATTER, run_format_output)
    graph_flow.add_node(SUMARIZER, run_sumarize)
    graph_flow.add_node(REFLECT, run_agent_reflection)
    graph_flow.add_node(ACT, safe_tool_node)
    graph_flow.add_conditional_edges(AGENT_REASON, should_continue, {
        END:END,
        ACT:ACT})
    graph_flow.add_edge(ACT, REFLECT)
    graph_flow.add_conditional_edges(REFLECT, after_reflect, {
        FORMATTER:FORMATTER, 
        SUMARIZER:SUMARIZER,
        AGENT_REASON:AGENT_REASON})
    graph_flow.add_edge(SUMARIZER, FORMATTER)
    graph_flow.add_edge(FORMATTER, END)
    sqlite_folder = Path("sqlite")
    sqlite_folder.mkdir(exist_ok=True)
    checkpoint_path = sqlite_folder / "checkpoints.sqlite"
    
    conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
    memory = SqliteSaver(conn)
    return graph_flow.compile(checkpointer=memory)

if __name__ == "__main__":
    graph = build_graph()
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    while True:
        user_text = input("Sobre qué querés hablar?: ")
        res = graph.invoke({"messages": [HumanMessage(content=user_text)]}, config=thread)
        print(clean_output(get_last_content(res["messages"])))
            