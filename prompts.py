def get_qualification_chat_prompt():
    with open("qualification_chat_prompt.txt", "r") as f:
        system_prompt = f.read()
    return system_prompt


def get_qualification_validate_prompt():
    with open("qualification_validation_prompt.txt", "r") as f:
        system_prompt = f.read()
    return system_prompt
