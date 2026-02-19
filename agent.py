import mimetypes
import os
import sys

from google import genai
from google.genai import types
from yaspin import yaspin
from yaspin.spinners import Spinners


class Agent:
    """A class to interact with the Google Gemini API."""

    def __init__(
        self, model="gemini-3-flash-preview", include_thoughts=False, temperature=0.2
    ):
        """
        Initialize the Agent with a Gemini API key.

        Args:
            api_key (str): The Google Gemini API key.
            model (str): The Gemini model to use (default: "gemini-3-flash-preview").
        """
        self.client = genai.Client()
        self._model = model
        # Global configuration for content generation (e.g., thinking config, temperature)
        self._config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=include_thoughts  # Global deaktiviert für Speed
            ),
            temperature=temperature,
        )
        self._pre_prompt = ""
        # check if the API key is set correctly in environment variable
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

    def set_pre_prompt(self, prompt):
        """
        Set a prompt to be prepended to all requests.

        Args:
            prompt (str): The prompt text to prepend to user input.
        """
        self._pre_prompt = prompt

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
                self._pre_prompt = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        except IOError as e:
            raise IOError(f"Error reading prompt file: {e}")

    @yaspin(Spinners.dots, text="AI is Thinking...", color="yellow", stream=sys.stderr)
    def ask(self, text):
        """
        Send a request to the Gemini API with optional pre-prompt.

        Args:
            text (str): The input text to send to the API.

        Returns:
            str: The response from the Gemini API.
        """
        full_text = self._pre_prompt + text if self._pre_prompt else text
        response = self.client.models.generate_content(
            model=self._model, contents=full_text, config=self._config
        )
        return response.text

    @yaspin(Spinners.dots, text="AI is Thinking...", color="green", stream=sys.stderr)
    def ask_with_files(self, text, file_paths):
        """
        Send a request to the Gemini API with file contents (supports PDFs, images, etc).

        Args:
            text (str): The input text/query to send to the API.
            file_paths (list): List of file paths to include (PDFs, images, text files, etc).

        Returns:
            str: The response from the Gemini API.
        """
        # Upload files and get file references
        uploaded_files = []
        for file_path in file_paths:
            try:
                # Detect mime type
                mime_type, _ = mimetypes.guess_type(file_path)
                if mime_type is None:
                    mime_type = "text/plain"

                # Upload the file using the file path directly
                response = self.client.files.upload(file=file_path)
                uploaded_files.append(response)
                # print(f"Uploaded: {file_path} (mime type: {mime_type})")
            except FileNotFoundError:
                print(f"Warning: File not found: {file_path}")
            except Exception as e:
                print(f"Error uploading {file_path}: {e}")

        # Prepare content with pre-prompt and files
        contents = [self._pre_prompt + text if self._pre_prompt else text]

        # Add uploaded files to contents
        for uploaded_file in uploaded_files:
            contents.append(uploaded_file)

        response = self.client.models.generate_content(
            model=self._model, contents=contents, config=self._config
        )
        return response.text
