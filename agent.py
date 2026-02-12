import os

from google import genai


class Agent:
    """A class to interact with the Google Gemini API."""

    def __init__(self):
        """
        Initialize the Agent with a Gemini API key.

        Args:
            api_key (str): The Google Gemini API key.
        """
        self.client = genai.Client()
        self.pre_prompt = ""
        # check if the API key is set correctly in environment variable
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

    def set_pre_prompt(self, prompt):
        """
        Set a prompt to be prepended to all requests.

        Args:
            prompt (str): The prompt text to prepend to user input.
        """
        self.pre_prompt = prompt

    def set_pre_prompt_from_file(self, file_path):
        """
        Set a prompt from a text file to be prepended to all requests.

        Args:
            file_path (str): The path to the text file containing the prompt.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If the file cannot be read.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                self.pre_prompt = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        except IOError as e:
            raise IOError(f"Error reading prompt file: {e}")

    def ask(self, text):
        """
        Send a request to the Gemini API with optional pre-prompt.

        Args:
            text (str): The input text to send to the API.

        Returns:
            str: The response from the Gemini API.
        """
        full_text = self.pre_prompt + text if self.pre_prompt else text
        response = self.client.models.generate_content(
            model="gemini-3-flash-preview", contents=full_text
        )
        return response.text


if __name__ == "__main__":
    # test 1: basic usage
    my_ai = Agent()
    my_ai.set_pre_prompt("You are a helpful assistant. ")
    response = my_ai.ask("What is the capital of France?")
    print(response)

    # test 2: set pre-prompt from file
    my_ai.set_pre_prompt_from_file("pre_prompt.txt")
    msg = ""
    with open("msg.txt", "r", encoding="utf-8") as f:
        msg = f.read()
    response = my_ai.ask(msg)
    print(response)
