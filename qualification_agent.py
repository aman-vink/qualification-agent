import asyncio
import os
from enum import Enum
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict, Any, Literal, Optional, TypedDict
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
import pprint
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser

from prompts import (
    get_qualification_chat_prompt,
    get_qualification_validate_prompt,
)
from qualification_process_state import QualificationProcessState

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class CategoryInternalValue(BaseModel):
    """
    Model representing the weight and value of a category.
    """

    name: str = Field(..., description="Name of the category value")
    weight: int = Field(..., description="Weight of the category value")


class Category(BaseModel):
    """
    Model representing a category with a unique identifier, human-readable name,
    value weights, overall weight, and the assigned value.
    """

    category_key: str = Field(..., description="Unique identifier for the category")
    category_name: str = Field(..., description="Human-readable name for the category")
    category_value_weights: List[CategoryInternalValue] = Field(
        ..., description="Mapping of internal values of the category"
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
    Model representing the full qualification configuration with weightage score settings.
    """

    weightage_score: WeightageScoreConfig = Field(
        ..., description="Configuration for weightage score based qualification"
    )


class QualificationResponse(BaseModel):
    distilled_chat: str = Field(
        ...,
        description="Distill the above chat messages into a single summary message. Include as many specific details as you can.",
    )
    ai_success_response: str = Field(
        ...,
        description="AI response if the qualification configuration is successfully updated.",
    )
    ai_failure_response: str = Field(
        ...,
        description="AI response if the qualification configuration is not successfully updated.",
    )


class QualificationConfigOutput(BaseModel):
    """
    Model representing the qualification configuration with weightage score settings.
    """

    operation_performed: Literal[
        QualificationProcessState.CATEGORY_CREATED,
        QualificationProcessState.CATEGORY_REMOVED,
        QualificationProcessState.CATEGORY_UPDATED,
        QualificationProcessState.CATEGORY_ADDED,
        QualificationProcessState.CATEGORY_INTERNAL_VALUES_REMOVED,
        QualificationProcessState.CATEGORY_INTERNAL_VALUES_ADDED,
        QualificationProcessState.CATEGORY_INTERNAL_VALUES_UPDATED,
        QualificationProcessState.CATEGORY_WEIGHT_UPDATED,
        QualificationProcessState.CATEGORY_WEIGHTS_UPDATED,
        QualificationProcessState.CATEGORY_WEIGHTS_CHANGED,
        QualificationProcessState.CATEGORY_INTERNAL_VALUE_WEIGHTS_CHANGED,
        QualificationProcessState.CATEGORY_NAME_CHANGED,
        QualificationProcessState.NO_CHANGE,
    ] = Field(..., description="Operation performed by user")
    qualification_config: QualificationDict = Field(
        ..., description="final qualification config after validation and changes"
    )


@tool("final_decision_tool")
def final_decision_tool(decision_str: str):
    """
    Tool to used when the user is ready to save the changes and AI gathered all the information required to change to the qualification configuration.
    """
    pprint.pprint(decision_str)
    return decision_str


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
        model="gpt-4o-mini",
        temperature=0.1,
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


async def summarize_messages(chat_messages):
    stored_messages = chat_messages
    if len(stored_messages) == 0:
        return []

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
    )

    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{chat_history}"),
            (
                "user",
                "Distill the above chat messages into a single summary message. Include as many specific details as you can.",
            ),
        ]
    )
    structured_llm = llm.with_structured_output(QualificationResponse)
    summarization_chain = summarization_prompt | structured_llm

    summary_message = await summarization_chain.ainvoke(
        {"chat_history": stored_messages}
    )
    return summary_message.dict()


def get_validation_chat_chain():
    """
    Creates and returns a chat chain for qualification using a prompt and a language model.
    """
    system_prompt = get_qualification_validate_prompt()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
        ]
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
    )
    structured_llm = llm.bind_tools([QualificationConfigOutput])
    validation_chain = (
        {
            "qualification_config": lambda x: x["qualification_config"],
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | structured_llm
    )
    return validation_chain


class QualificationAgentState(TypedDict):
    """
    State model for the qualification agent containing user input, chat history,
    qualification configuration, and AI output.
    """

    user_input: str
    chat_history_db: BaseChatMessageHistory
    chat_history: list[BaseMessage]
    qualification_config: QualificationDict
    ai_output: str
    qualification_process_state: Optional[QualificationProcessState]


async def get_validated_qualification_config(chat_history, qualification_config):
    validation_chain = get_validation_chat_chain()
    agent_output = await validation_chain.ainvoke(
        {
            "qualification_config": qualification_config,
            "chat_history": chat_history,
        }
    )

    if isinstance(agent_output, QualificationConfigOutput):
        return agent_output.dict()

    tool_calls = agent_output.tool_calls
    if not tool_calls:
        print("No tool calls")

    qualification_config = tool_calls[0]["args"]
    return qualification_config


async def output_state(state):
    user_and_ai_chat = state["chat_history"]

    chat_history = []

    # no need to pass tool msg in chat history.
    first_ai_message = True
    for msg in user_and_ai_chat:
        if isinstance(msg, AIMessage) and (msg.tool_calls or first_ai_message):
            if first_ai_message:
                first_ai_message = False
            continue
        chat_history.append(msg)

    print("OYOYOOYOY")

    current_qualification_config = state["qualification_config"]
    tasks = []
    tasks.append(
        get_validated_qualification_config(chat_history, current_qualification_config)
    )
    tasks.append(summarize_messages(chat_history))
    results = await asyncio.gather(*tasks)
    return results


def final_decision_node(state):
    """
    Updates the qualification configuration in the state with the latest content
    from the chat history.
    """
    results = asyncio.run(output_state(state))
    summary_output = results[1]
    qualification_config_output = results[0]

    print(summary_output)
    print(qualification_config_output)

    operation_performed = qualification_config_output["operation_performed"]
    state["qualification_process_state"] = operation_performed
    state["qualification_config"] = qualification_config_output["qualification_config"]
    state["ai_output"] = summary_output["ai_success_response"]

    ai_output = state["ai_output"]
    chat_history = state["chat_history"]

    # remove the previous chat history and add the distilled chat message.
    new_chat_history = []

    first_ai_message = True
    for msg in chat_history:
        if isinstance(msg, SystemMessage):
            new_chat_history.append(msg)

    new_chat_history.append(AIMessage(summary_output["distilled_chat"]))
    print(chat_history)
    state["chat_history"] = new_chat_history
    state["chat_history_db"].add_ai_message(AIMessage(ai_output))
    return state


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
    tool_calls_list = agent_output.tool_calls

    # no tools called.
    if not tool_calls_list:
        return False

    # we have decided to use one tool only so it should be final decision tool.
    tool_dict = tool_calls_list[0]

    tool_name = tool_dict.get("name")

    print(tool_name)
    # if tool_name == "adjust_categories_weight_tool":
    #     print(output)
    #     rearrange_chain = get_rearrange_categories_weight_tool_chain()
    #     output = rearrange_chain.invoke(
    #         {"categories_weights_message": state["chat_history"][-1].content}
    #     )
    if tool_name != "final_decision_tool":
        return False

    # original qualification config passed.
    qualification_config = tool_dict.get("args")
    print(qualification_config)
    return True


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
        "chat",
        validate_step_node,
        {False: "output", True: "final_changes"},
    )
    graph.add_edge("final_changes", END)
    graph.add_edge("output", END)

    graph.set_entry_point("chat")
    app = graph.compile()
    return app
