#! /usr/bin/env python3

from pprint import pprint as pp
from agent import Agent
import mimetypes
import re

mimetypes.add_type('text/plain', '.log')
mimetypes.add_type('text/plain', '.LOG')

def print_colored_response(response_text):
    """Prints the response, interpreting ANSI color codes.

    If the API returns escaped sequences like "\\u001b[31m", we
    decode them so the terminal shows colours. Do not touch ordinary
    Unicode characters (ä, ö, ü etc) unless an escape is present.
    """
    # decode only when there are literal escape sequences
    if "\\u" in response_text or "\\x1b" in response_text:
        try:
            response_text = response_text.encode("utf-8").decode("unicode_escape")
        except Exception:
            pass
    print(response_text, end="")

# helper used when output is not a tty; strips ANSI codes entirely
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from a string."""
    return _ANSI_ESCAPE_RE.sub("", text)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="P(ython)AI(A)gent CLI - Interact with the Google Gemini API from the command line."
    )
    parser.add_argument(
        "-p",
        "--pre-prompt",

        type=str,
        default=None,
        help="File containing pre-prompt text.",
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        default=None,
        help="File to send with the prompt (supports wildcards), -- to indicate the end of options and the start of the prompt. Example: --files file1.txt file2.pdf -- 'What is in these files?'",
    )
    parser.add_argument("prompt", type=str, help="Prompt to send to the agent.")
    args = parser.parse_args()

    # pp(vars(args))
    # exit(0)

    # Initialize the agent with the specified model
    agent = Agent(model="gemini-2.5-flash", temperature=1)

    # Set pre-prompt if provided
    if args.pre_prompt:
        agent.set_pre_prompt_from_file(args.pre_prompt)

    # Send the prompt with or without files based on the arguments
    if args.files:
        response = agent.ask_with_files(args.prompt, args.files)
    else:
        response = agent.ask(args.prompt)

    if sys.stdout.isatty():
        print_colored_response(response)
    else:
        # remove any ANSI colour codes when output is redirected or piped
        print(strip_ansi(response))
