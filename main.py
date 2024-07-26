from qualification_agent import create_graph
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

msgs = StreamlitChatMessageHistory()
msgs.add_ai_message("Should I remove buyer type?")
app = create_graph()
output = app.invoke({
                "user_input": "Yes remove the buyer type.",
                "chat_history": msgs,
                "qualification_config": get_,
                "ai_output": "",
            })

# app.invoke()

print(output["chat_history"].messages)

print(output)