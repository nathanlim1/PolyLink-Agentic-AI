import json
import operator
from typing import Literal, Optional, Union
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing_extensions import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv()
MODEL = "gpt-4o-mini"


# Definition of the shared state schema (used to communicate and share data between nodes)
class State(TypedDict):
    user_query: str             # Initial user query
    user_data: dict             # User metadata
    final_answer: str           # Final answer to be displayed to the user
    handoff: str                # If non-empty, indicates handoff of final_answer to specific node
    sub_agent_prompts: dict     # Prompts for sub-agents {"general": "...", "clubs": "...", etc.}
    aggregate_answers: Annotated[list, operator.add]  # The operator.add reducer fn makes this append-only

# Define the output schema for the central agent
class CentralAgentOutput(BaseModel):
    final_answer: str = Field(None, description="The final answer to the user's query (blank if insufficient info)")
    handoff: str = Field(None, description="'general', 'professor', 'clubs', or 'schedule' if indicating handoff")
    general_prompt: str = Field(None, description="If a call is needed, prompt for the general assistant")
    clubs_prompt: str = Field(None, description="If a call is needed, prompt for the clubs assistant")
    professor_prompt: str = Field(None, description="If a call is needed, prompt for the professor assistant")
    schedule_prompt: str = Field(None, description="If a call is needed, prompt for the schedule assistant")


# Initialization of LLM models with proper tool bindings for each node
# TODO: replace these to use specific prompt-engineered preset assistants from our OpenAI account.
central_llm = ChatOpenAI(model=MODEL).with_structured_output(CentralAgentOutput)
general_llm = ChatOpenAI(model=MODEL)
clubs_llm = ChatOpenAI(model=MODEL)
professor_llm = ChatOpenAI(model=MODEL)
prof_parser_assistant = ChatOpenAI(model=MODEL)
schedule_llm = ChatOpenAI(model=MODEL)
schedule_parser_assistant = ChatOpenAI(model=MODEL)

print(f"[LOG] All LLM assistants with model {MODEL} initialized.")

def query_prof_db(query: str) -> str:
    # TODO: Implement the actual database query logic here
    return "EXAMPLE PROFESSOR DATA: none"

def query_schedule_db(query: str) -> str:
    # TODO: Implement the actual database query logic here
    return "EXAMPLE SCHEDULE DATA: none"


def central_agent(state: State) -> dict:
    """NODE 1 - Central Agent"""
    print("[LOG] Entering central_agent node.")
    collected_info = ", ".join(state["aggregate_answers"])

    if not collected_info:
        # No information collected yet means that this is the initial prompt to the central agent
        decision_prompt = (
            f"User Query: {state['user_query']}\n\n"
            "You are the Central Agent responsible for managing a team of four specialized sub-agents. Your goal is to "
            "determine whether the user's query can be answered directly by delegating to a single sub-agent or if it "
            "requires a coordinated, multi-step approach. Use the following guidelines:\n\n"
            "1. Sub-agent Capabilities:\n"
            "   - General Assistant: Possesses in-depth knowledge about Cal Poly, including historical context and "
            "     current campus details, and can conduct web searches.\n"
            "   - Clubs Assistant: Retrieves, analyzes, and summarizes data regarding Cal Poly clubs and organizations.\n"
            "   - Professor Ratings Assistant: Fetches and evaluates professor review data to provide insights into teaching quality.\n"
            "   - Schedule Analysis Assistant: Examines user-provided schedule information alongside Cal Poly course data to generate "
            "     effective recommendations or conflict analyses.\n\n"
            "2. Decision Process:\n"
            "   - If the query is straightforward, identify the most relevant sub-agent and simply handoff the task.\n"
            "   - If the query is complex or multi-faceted, issue clear and detailed instructions to the appropriate sub-agents"
            "     to gather the necessary information you need to answer. If this is the case, leave handoff blank.\n\n"
        )
    else:
        # If there is already some information collected, see if it's sufficient to answer the query
        decision_prompt = (
            f"User Query: {state['user_query']}\n\n"
            f"Collected Information So Far:\n{collected_info}\n\n"
            "Your task is twofold: first, evaluate whether the current information fully addresses the user's query. "
            "If it does, provide a concise, comprehensive final answer. If it does not, identify the missing elements and issue "
            "specific, step-by-step instructions to the appropriate sub-agent(s) to gather additional data.\n\n"
            "Sub-agent Capabilities:\n"
            "   - General Assistant: Possesses in-depth knowledge about Cal Poly, including historical context and "
            "     current campus details, and can conduct web searches.\n"
            "   - Clubs Assistant: Retrieves, analyzes, and summarizes data regarding Cal Poly clubs and organizations.\n"
            "   - Professor Ratings Assistant: Fetches and evaluates professor review data to provide insights into teaching quality.\n"
            "   - Schedule Analysis Assistant: Examines user-provided schedule information alongside Cal Poly course data to generate "
            "     effective recommendations or conflict analyses.\n\n"
            "Guidelines for Your Response:\n"
            "   1. Evaluate the current information thoroughly to determine if it fully answers the user's query.\n"
            "   2. If sufficient, provide a well-structured summary answer that incorporates all relevant details.\n"
            "   3. If additional information is needed, clearly specify what is missing and delegate precise, targeted tasks to "
            "      the relevant sub-agent(s).\n\n"
        )

    print(f"[LOG] Sending decision_prompt to central_llm:\n{decision_prompt}")
    decision_response = central_llm.invoke(decision_prompt)
    print(f"[LOG] central_llm decision response: {decision_response}")

    return {
        "final_answer": decision_response.final_answer,
        "handoff": decision_response.handoff,
        "sub_agent_prompts": {
            "general": decision_response.general_prompt,
            "clubs": decision_response.clubs_prompt,
            "professor": decision_response.professor_prompt,
            "schedule": decision_response.schedule_prompt
        }
    }


def general_assistant(state: State) -> dict:
    """NODE 2 - General Assistant Node"""
    print("[LOG] Entering general_assistant node.")
    if state["handoff"] == "general":
        # Handoff -> use the user's query as the prompt
        prompt = state["user_query"]
    else:
        prompt = state["sub_agent_prompts"]["general"]

    print(f"[LOG] Using general_prompt: {prompt}")
    response = general_llm.invoke(prompt).content.strip()
    print(f"[LOG] LLM response from general_assistant: {response}")

    if state["handoff"] == "general":
        # If this is a handoff, update final answer rather than adding answer to aggregate
        return {"final_answer": response}
    else:
        return {"aggregate_answers": [response]}


def clubs_assistant(state: State) -> dict:
    """NODE 2 - Clubs Assistant Node"""
    print("[LOG] Entering clubs_assistant node.")
    if state["handoff"] == "clubs":
        # Handoff -> use the user's query as the prompt
        prompt = state["user_query"]
    else:
        prompt = state["sub_agent_prompts"]["clubs"]
    print(f"[LOG] Using clubs_prompt: {prompt}")
    response = clubs_llm.invoke(prompt).content.strip()
    print(f"[LOG] LLM response from clubs_assistant: {response}")

    if state["handoff"] == "clubs":
        # If this is a handoff, update final answer rather than adding answer to aggregate
        return {"final_answer": response}
    else:
        return {"aggregate_answers": [response]}


def professor_ratings(state: State) -> dict:
    """NODE 3 - Professor Ratings and Courses Node"""
    print("[LOG] Entering professor_ratings node.")
    if state["handoff"] == "professor":
        # Handoff -> use the user's query as the prompt
        prompt = state["user_query"]
    else:
        prompt = state["sub_agent_prompts"]["professor"]

    # parser assistant builds query
    query = prof_parser_assistant.invoke(prompt).content.strip()

    # query the database for the results
    results = query_prof_db(query)

    prompt += "\nFetched data to answer user query: " + results

    print(f"[LOG] Using professor_prompt: {prompt}")
    response = professor_llm.invoke(prompt).content.strip()
    print(f"[LOG] LLM response from professor_ratings: {response}")

    if state["handoff"] == "professor":
        # If this is a handoff, update final answer rather than adding answer to aggregate
        return {"final_answer": response}
    else:
        return {"aggregate_answers": [response]}


def schedule_analysis(state: State) -> dict:
    """Node 4 - Schedule Analysis Node"""
    print("[LOG] Entering schedule_analysis node.")
    if state["handoff"] == "schedule":
        # Handoff -> use the user's query as the prompt
        prompt = state["user_query"]
    else:
        prompt = state["sub_agent_prompts"]["schedule"]

    # parser assistant builds query
    query = schedule_parser_assistant.invoke(prompt).content.strip()

    # query the database for the results
    results = query_schedule_db(query)

    prompt = prompt + "\nUse the following fetched data:" + results

    print(f"[LOG] Using schedule_prompt: {prompt}")
    response = schedule_llm.invoke(prompt).content.strip()
    print(f"[LOG] LLM response from schedule_analysis: {response}")

    if state["handoff"] == "schedule":
        # If this is a handoff, update final answer rather than adding answer to aggregate
        return {"final_answer": response}
    else:
        return {"aggregate_answers": [response]}


def route_from_central(state: State) -> list[str]:
    """Routing function for conditional edges from the central agent node"""
    if state["final_answer"]:
        return END
    else:
        parallel_calls = []
        # Check which sub-agents need to be called
        if state["sub_agent_prompts"]["general"]:
            parallel_calls.append("general_assistant")
        if state["sub_agent_prompts"]["clubs"]:
            parallel_calls.append("clubs_assistant")
        if state["sub_agent_prompts"]["professor"]:
            parallel_calls.append("professor_ratings")
        if state["sub_agent_prompts"]["schedule"]:
            parallel_calls.append("schedule_analysis")
        return parallel_calls

def route_from_sub_asst(state: State) -> str:
    """Routing function for conditional edges from the sub-assistant nodes.
    If a handoff was initiated, simply route to the END; otherwise, return to the central agent."""
    if state["handoff"] and state["final_answer"]:
        return END
    else:
        return "central_agent"


# STATE GRAPH CONSTRUCTION
graph = StateGraph(State)
graph.add_node("central_agent", central_agent)
graph.add_node("general_assistant", general_assistant)
graph.add_node("clubs_assistant", clubs_assistant)
graph.add_node("professor_ratings", professor_ratings)
graph.add_node("schedule_analysis", schedule_analysis)

# Start with the central agent node.
graph.add_edge(START, "central_agent")

# Conditional edges from the central agent to either go to sub-agents or to END (if final answer was constructed).
graph.add_conditional_edges("central_agent", route_from_central)

# Conditional edges from each sub-agent to either go back to the central agent or to END (if handoff was initiated).
graph.add_conditional_edges("general_assistant", route_from_sub_asst)
graph.add_conditional_edges("clubs_assistant", route_from_sub_asst)
graph.add_conditional_edges("professor_ratings", route_from_sub_asst)
graph.add_conditional_edges("schedule_analysis", route_from_sub_asst)


compiled_graph = graph.compile()

# Example invocation:
initial_state: State = {
    "user_query": "Can you tell me about how well my clubs align with my current schedule?",
    "user_data": {
        "user_id": "12345",
        "user_metadata": {
            "interests": ["technology", "sports"],
            "year": "junior"
        }
    },
    "final_answer": "",
    "handoff": "",
    "sub_agent_prompts": {
        "general": "",
        "clubs": "",
        "professor": "",
        "schedule": ""
    },
    "aggregate_answers": []
}
print("[LOG] Invoking compiled graph with initial state:")
print(initial_state)

final_state = compiled_graph.invoke(initial_state)
print("[LOG] Final state received from graph invocation.")
print("Final Answer:", final_state["final_answer"])
