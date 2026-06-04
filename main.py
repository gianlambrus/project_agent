import os 
from pathlib import Path
import sqlite3
from typing import TypedDict
from dotenv import load_dotenv
from graph.nodes import run_agent_reasoning, tool_node
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessagesState, StateGraph, END

load_dotenv()

AGENT_REASON="agent_reason"
ACT="act"
LAST=-1

def should_continue(state:MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

graph_flow = StateGraph(MessagesState)

graph_flow.add_node(AGENT_REASON, run_agent_reasoning)
graph_flow.set_entry_point(AGENT_REASON)
graph_flow.add_node(ACT, tool_node)
graph_flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END:END,
    ACT:ACT})
graph_flow.add_edge(ACT, AGENT_REASON)

sqlite_folder = Path("sqlite")
sqlite_folder.mkdir(exist_ok=True)
checkpoint_path = sqlite_folder / "checkpoints.sqlite"

with SqliteSaver.from_conn_string(str(checkpoint_path)) as memory:
    
    graph = graph_flow.compile(checkpointer=memory)
    graph.get_graph().draw_mermaid_png(output_file_path="graph_flow.png")

    if __name__ == "__main__":
        thread = {"configurable": {"thread_id": "777"}}
        
        initial_input = {"input": "test"}
        
        for event in graph.stream(initial_input, thread, stream_mode="values"):
            print(event)
        
        print(graph.get_state(thread).next)    
                
        print(graph.get_state(thread))
        
        print(graph.get_state(thread).next)
        
        for event in graph.stream(input=None, config=thread, stream_mode="values"):
            print(event)
    
        while True:
            user_text = input("Sobre qué querés hablar?: ")
            res = graph.invoke({"messages": [HumanMessage(content=user_text)]},
                    config=thread)
            print(res["messages"][LAST].content)