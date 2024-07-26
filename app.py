from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import json
import pandas as pd
import streamlit as st

from qualification_agent import create_graph

import os
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "agent"

def get_qualification_table(qualification_data):
    """
    Creates a DataFrame with the qualification data for displaying in a table.
    
    Args:
        qualification_data (dict): The qualification data.
    
    Returns:
        pd.DataFrame: The DataFrame containing the qualification data.
    """
    data = []
    for category in qualification_data["type_config"]["weightage_score"]["categories"]:
        category_name = category["category_name"]
        category_weight = category["category_weight"]
        category_values = "\n".join(
            f"{value} ({weight})"
            for value, weight in category["category_value_weights"].items()
        )
        data.append([category_name, category_weight, category_values])

    # Creating a DataFrame
    df = pd.DataFrame(
        data, columns=["Category Name", "Category Weight", "Category Values (Weight)"]
    )
    return df

def display_qualification_table(qualification_pd_table):
    """
    Displays the qualification table in the Streamlit sidebar.
    
    Args:
        qualification_pd_table (pd.DataFrame): The DataFrame containing the qualification data.
    """
    st.sidebar.write(
        qualification_pd_table.to_html(escape=False, index=False).replace("\\n", "<br>"),
        unsafe_allow_html=True,
    )

def get_qualification_weightage_table(qualification_json):
    """
    Extracts the weightage table from the qualification JSON data.
    
    Args:
        qualification_json (dict): The qualification JSON data.
    
    Returns:
        dict: The weightage table.
    """
    return qualification_json["type_config"]["weightage_score"]["categories"]

# Load the qualification configuration from a JSON file
with open("qualification_config.json", "r") as f:
    qualification_config = json.load(f)

qualification_df = get_qualification_table(qualification_config)

st.set_page_config(page_title="Qualification Agent", page_icon="🦜")

st.title("🦜Vink Qualification Agent")
# Show table in sidebar
st.sidebar.title("Qualification Table")

display_qualification_table(qualification_df)

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

msgs = StreamlitChatMessageHistory()

memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="output",
)

if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    st.session_state["chat_history"]= []

    with st.chat_message("ai"):
        st.write(
            """
            Hi there! Welcome to Vink.ai's qualification scoring system manager. I'm here to help you tweak your sales pipeline qualifications. You can:

            1. **Add a New Category:** 
            - Tell me the name of the new category.
            - I’ll suggest some values and weights, and you can adjust them as needed.
            - Let me know how important this category is compared to others.

            2. **Update an Existing Category:** 
            - Pick a category to update.
            - Change its values and weights.

            3. **Change Qualification Criteria:**
            - Let me know what top-level changes you need.

            4. **Add Multiple Categories:**
            - We can go through each new category one by one.

            5. **Remove a Category:** 
            - Tell me which category you want to remove.

            Just let me know what you’d like to do, and we’ll get started!
            """
        )
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        st.write(msg.content)

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.chat_message("user").write(prompt)

    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()
    app = create_graph()
    chat_history = st.session_state.get("chat_history")
    if not st.session_state.get("chat_history"):
        st.session_state["chat_history"] = []

    with st.chat_message("assistant"):
        response = app.invoke(
            {
                "user_input": prompt,
                "chat_history_db": msgs,
                "chat_history": st.session_state["chat_history"],
                "qualification_config": get_qualification_weightage_table(qualification_config),
                "ai_output": "",
            }
        )
        ai_output = response["ai_output"]
        st.session_state["chat_history"].extend(response["chat_history"])
        st.write(response["ai_output"])
