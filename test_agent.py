import sys
from agent import Agent



def test_agent(model=None):
    print(f"Testing Agent with model: {model}")
    # test 1: basic usage
    print("_________________ TEST 1 __________________")
    my_ai = Agent(model)
    response = my_ai.ask("What is the capital of France?")
    print(response)
    print("")

    # test 2: no pre-prompt
    print("_________________ TEST 2 __________________")
    my_ai.set_pre_prompt("You are a helpful only french speaking assistant. ")
    response = my_ai.ask("What is the capital of Germany?")
    print(response)
    print("")

    # test 3: set pre-prompt from file
    print("_________________ TEST 3 __________________")
    my_ai.set_pre_prompt_from_file("pre_prompt.txt")
    msg = ""
    with open("msg.txt", "r", encoding="utf-8") as f:
        msg = f.read()
    response = my_ai.ask_with_files(msg, ["lebenslauf.txt"])
    print(response)
    print("")

    # test 4: chat history from file
    print("_________________ TEST 4 __________________")
    my_ai.set_pre_prompt_from_file("pre_prompt.txt")
    msg = ""
    with open("chat.txt", "r", encoding="utf-8") as f:
        msg = f.read()
    response = my_ai.ask(msg)
    print(response)
    print("")

    # test 5: ask with files (PDF, image, text)
    print("_________________ TEST 5 __________________")
    my_ai.set_pre_prompt_from_file("pre_prompt.txt")
    with open("chat.txt", "r", encoding="utf-8") as f:
        msg = f.read()
    response = my_ai.ask_with_files(msg, ["lebenslauf.txt"])
    print(response)
    print("")

if __name__ == "__main__":
    test_agent("gemini-2.5-flash")
    test_agent("gemini-2.5-pro")
