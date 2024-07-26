from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict, Any, TypedDict
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
import pprint

from prompts import get_qualification_chat_prompt


class Category(BaseModel):
    """
    Model representing a category with a unique identifier, human-readable name, 
    value weights, overall weight, and the assigned value.
    """
    category_key: str = Field(..., description="Unique identifier for the category")
    category_name: str = Field(..., description="Human-readable name for the category")
    category_value_weights: Dict[str, int] = Field(
        ..., description="Mapping of category values to their respective weights"
    )
    category_weight: int = Field(..., description="Weight of the category")
    value: str = Field(..., description="Value assigned to the category")


class WeightageScoreConfig(BaseModel):
    """
    Model representing the configuration for weightage scores with a list of categories.
    """
    categories: List[Category] = Field(
        ..., description="List of categories with their configurations"
    )


class QualificationDict(BaseModel):
    """
    Model representing the qualification configuration with weightage score settings.
    """
    weightage_score: WeightageScoreConfig = Field(
        ..., description="Configuration for weightage score based qualification"
    )


@tool("final_decision_tool", args_schema=QualificationDict)
def final_decision_tool(qualification_config: QualificationDict):
    """
    Tool to change the qualification configuration after user confirmation.
    """
    pprint.pprint(qualification_config)
    return qualification_config


def get_qualification_chat_chain():
    """
    Creates and returns a chat chain for qualification using a prompt and a language model.
    """
    system_prompt = get_qualification_chat_prompt()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        openai_api_key="sk-proj-e50r9tVVGRDWwCO4mwRFT3BlbkFJlIPc0EJoXdvF6fdhpKkM",
    )

    llm = llm.bind_tools([final_decision_tool])

    qualification_chain = (
        {
            "qualification_config": lambda x: x["qualification_config"],
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm
    )
    return qualification_chain


class QualificationAgentState(TypedDict):
    """
    State model for the qualification agent containing user input, chat history, 
    qualification configuration, and AI output.
    """
    user_input: str
    chat_history_db: BaseChatMessageHistory
    chat_history: list[BaseMessage]
    qualification_config: dict[str, Any]
    ai_output: str


def final_decision_node(state):
    """
    Updates the qualification configuration in the state with the latest content 
    from the chat history.
    """
    print("Executing final_decision_node")
    qualification_data = state["chat_history"][-1].content
    state["qualification_config"] = qualification_data


def agent_output_node(state):
    """
    Sets the AI output in the state with the latest content from the chat history.
    """
    print("Executing agent_output_node")
    state["ai_output"] = state["chat_history"][-1].content
    return state


def validate_step_node(state):
    """
    Validates whether the agent used the final decision tool in the last tool call.
    """
    print("Executing validate_step_node")
    agent_output = state["chat_history"][-1]
    print(agent_output)
    if agent_output.type == "tool":
        tool_calls = agent_output.tool_calls
        if len(tool_calls) == 1 and tool_calls[0].get("name") == "final_decision_tool":
            return True
    return False


def chat_node(state):
    """
    Processes user input, updates chat history, and invokes the qualification chain.
    """
    print("Executing chat_node")
    user_input = state["user_input"]
    chat_history_db = state["chat_history_db"]
    chat_history = state["chat_history"]
    chat_history_db.add_user_message(HumanMessage(user_input))
    qualification_config = state["qualification_config"]

    qualification_chain = get_qualification_chat_chain()
    agent_output = qualification_chain.invoke(
        {
            "input": user_input,
            "chat_history": state["chat_history"],
            "qualification_config": qualification_config,
        }
    )
    chat_history_db.add_ai_message(AIMessage(agent_output.content))
    chat_history.append(HumanMessage(user_input))
    chat_history.append(agent_output)
    state["chat_history"] = chat_history
    return state


def create_graph():
    """
    Creates and returns a state graph for the qualification agent with nodes for chat, 
    final decision, and output, along with conditional edges for validation.
    """
    print("Creating graph")
    graph = StateGraph(QualificationAgentState)
    graph.add_node("chat", chat_node)
    graph.add_node("output", agent_output_node)
    graph.add_node("final_changes", final_decision_node)

    graph.add_conditional_edges(
        "chat", validate_step_node, {False: "output", True: "final_changes"}
    )
    graph.add_edge("final_changes", END)
    graph.add_edge("output", END)

    graph.set_entry_point("chat")
    app = graph.compile()
    return app
