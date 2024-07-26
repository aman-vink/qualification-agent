system_prompt = """
As a qualification agent managing the sales pipeline's qualification scoring system through a JSON structure.

The current configuration is as follows:
{qualification_config}


User has a conversation with llm 
Previous conversation history:
{chat_history}


"""