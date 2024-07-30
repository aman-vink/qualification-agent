from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import json
import pandas as pd
import streamlit as st

from qualification_agent import create_graph
import os
from qualification_process_state import QualificationProcessState

st.set_page_config(page_title="Qualification Agent", page_icon="🦜")


os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def input_json(qualification_config):
    """Function to input a JSON string and load its content."""
    json_input = st.sidebar.text_area(
        "Enter Qualification Config JSON",
        height=200,
        value=json.dumps(
            st.session_state.get("temp_qualification_config", qualification_config),
            indent=4,
        ),
        key="json_input",
    )
    if st.sidebar.button("Submit JSON"):
        try:
            config = json.loads(json_input)
            st.session_state["qualification_config"] = config
            st.session_state["temp_qualification_config"] = config
            st.session_state["qualification_df"] = get_qualification_table(config)
            st.sidebar.success("JSON updated successfully!")
        except json.JSONDecodeError:
            st.error("Error: Invalid JSON format. Please correct and try again.")
    return None


def get_qualification_table(qualification_data):
    """Creates a DataFrame with the qualification data for displaying in a table."""
    if (
        not qualification_data
        or "type_config" not in qualification_data
        or "weightage_score" not in qualification_data["type_config"]
        or "categories" not in qualification_data["type_config"]["weightage_score"]
    ):
        return pd.DataFrame(
            columns=["Category Name", "Category Weight", "Category Values (Weight)"]
        )

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


def update_qualification_table(updated_categories, qualification_json):
    """Updates the qualification table with the new qualification data."""
    if not updated_categories:
        st.error(
            "No updated categories provided. Cannot update the qualification config."
        )
        return

    for category in updated_categories:
        category_value_weights = category.get("category_value_weights", [])
        category_value_weights = {
            category_value_weight.get("name"): category_value_weight.get("weight")
            for category_value_weight in category_value_weights
        }
        category["category_value_weights"] = category_value_weights

    qualification_json["type_config"]["weightage_score"][
        "categories"
    ] = updated_categories

    try:
        with open("qualification_config.json", "w") as f:
            json.dump(qualification_json, f, indent=4)
    except Exception as e:
        st.error(f"Error saving qualification configuration: {e}")


def display_qualification_table(qualification_pd_table):
    """Displays the qualification table in the Streamlit sidebar."""
    st.sidebar.write(
        qualification_pd_table.to_html(escape=False, index=False).replace(
            "\\n", "<br>"
        ),
        unsafe_allow_html=True,
    )


def get_qualification_weightage_table(qualification_json):
    """Extracts the weightage table from the qualification JSON data."""
    return qualification_json["type_config"]["weightage_score"]["categories"]


def load_qualification_config():
    with open("default_qualification_config.json", "r") as f:
        config = json.load(f)
    return config


def load_prompt(file_name):
    with open(file_name, "r") as f:
        return f.read()


def save_prompt(file_name, prompt):
    with open(file_name, "w") as f:
        f.write(prompt)


def download_json(data, file_name):
    json_str = json.dumps(data, indent=4)
    st.sidebar.download_button(
        label="Download JSON",
        data=json_str,
        file_name=file_name,
        mime="application/json",
    )


def reset_prompt(file_name, text_area_key):
    """Resets the prompt to its default value."""
    default_prompt = load_prompt(file_name)
    st.session_state[text_area_key] = default_prompt
    save_prompt(file_name, default_prompt)


default_qualification_config = load_qualification_config()
default_qualification_chat_prompt = load_prompt("default_qualification_chat_prompt.txt")
default_qualification_validation_prompt = load_prompt(
    "default_qualification_validation_prompt.txt"
)


def display_qualification_state():
    """Displays the current state of the qualification table."""
    st.sidebar.write(
        "Qualification Table Current State:",
        st.session_state.get("qualification_process_state", "Not started"),
    )
    display_qualification_table(st.session_state["qualification_df"])


def update_session_state(qualification_process_state, qualification_df):
    """Updates the session state with the qualification process state and DataFrame."""
    st.session_state["qualification_process_state"] = qualification_process_state
    st.session_state["qualification_df"] = qualification_df


st.title("🦜Vink Qualification Agent")
st.sidebar.title("Qualification Table")

if "qualification_config" not in st.session_state:
    st.session_state["qualification_config"] = default_qualification_config

input_json(st.session_state["qualification_config"])

with open("qualification_config.json", "w") as f:
    json.dump(st.session_state["qualification_config"], f, indent=4)

qualification_df = get_qualification_table(st.session_state["qualification_config"])
st.session_state["qualification_df"] = qualification_df
display_qualification_table(qualification_df)

# Download JSON button
download_json(st.session_state["qualification_config"], "qualification_config.json")

# Prompt editing
chat_prompt_key = "chat_prompt"
validation_prompt_key = "validation_prompt"

if chat_prompt_key not in st.session_state:
    st.session_state[chat_prompt_key] = default_qualification_chat_prompt

if validation_prompt_key not in st.session_state:
    st.session_state[validation_prompt_key] = default_qualification_validation_prompt

chat_prompt = st.sidebar.text_area(
    "Edit Qualification Chat Prompt",
    value=st.session_state.get(chat_prompt_key, default_qualification_chat_prompt),
    height=200,
    key=chat_prompt_key,
)

if st.sidebar.button("Save Chat Prompt"):
    save_prompt("qualification_chat_prompt.txt", chat_prompt)
    st.sidebar.success("Chat prompt saved successfully!")

validation_prompt = st.sidebar.text_area(
    "Edit Qualification Validation Prompt",
    value=st.session_state.get(
        validation_prompt_key, default_qualification_validation_prompt
    ),
    height=200,
    key=validation_prompt_key,
)

if st.sidebar.button("Save Validation Prompt"):
    save_prompt("qualification_validation_prompt.txt", validation_prompt)
    st.sidebar.success("Validation prompt saved successfully!")

msgs = StreamlitChatMessageHistory()

memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="output",
)

if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    st.session_state["chat_history"] = []

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
                "qualification_config": get_qualification_weightage_table(
                    st.session_state["qualification_config"]
                ),
                "qualification_process_state": QualificationProcessState.CONVERSATION_INITIATED,
                "ai_output": "",
            }
        )

        qualification_process_state = response["qualification_process_state"]
        if qualification_process_state not in [
            QualificationProcessState.ERROR.value,
            QualificationProcessState.NO_CHANGE.value,
            QualificationProcessState.CONVERSATION_INITIATED.value,
        ]:
            updated_categories = (
                response["qualification_config"]
                .get("weightage_score", {})
                .get("categories", [])
            )
            update_qualification_table(
                updated_categories, st.session_state["qualification_config"]
            )

            st.session_state["qualification_config"] = load_qualification_config()
            qualification_df = get_qualification_table(
                st.session_state["qualification_config"]
            )
            update_session_state(qualification_process_state, qualification_df)
            display_qualification_state()

        st.session_state["chat_history"] = response["chat_history"]

        st.write(response["ai_output"])
