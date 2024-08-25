import json
import re
import requests
from collections.abc import Callable
from typing import Annotated as A, Literal as L
from pydantic import BaseModel
import datetime
import openai

client = openai.OpenAI()


class StopException(Exception):
    """
    Stop Execution by raising this exception (Signal that the task is Finished).
    """


def supatool_search(searchQuery: str) -> str:
    """
    Search for new functionality (tools) to extend your current capabilities`.
    """
    # empty the dynamic toolbox
    dynamic_toolbox = std_lib.copy()

    # add new tools from the search
    url = "http://localhost:8000/supatool/v1/search"
    payload = {"query": searchQuery}
    response = requests.post(url, json=payload)
    data = response.json()
    endpoints = data[0]["endpoints"]
    for tool in endpoints:
        tool_json = json.loads(tool["toolString"])
        dynamic_toolbox.append(tool_json)
        # keep a mapping of the tool names to their cuids
        # so the model doesnt have to keep track
        supatool_name_to_cuid_map[tool["name"]] = tool["cuid"]

    return json.dumps(response.json())


def supatool_execute(cuid: str, params: dict[str, any]) -> str:
    url = "http://localhost:8000/supatool/v1/execute/" + cuid
    payload = {"params": params}
    print(payload)
    response = requests.post(url, json=payload)
    return json.dumps(response.json())


def finish(answer: A[str, "Answer to the user's question."]) -> None:
    """Answer the user's question, and finish the conversation."""
    raise StopException(answer)


def get_current_location() -> str:
    """Get the current location of the user."""
    return json.dumps(requests.get("http://ip-api.com/json?fields=lat,lon").json())


def calculate(
    formula: A[str, "Numerical expression to compute the result of, in Python syntax."],
) -> str:
    """Calculate the result of a given formula."""
    return str(eval(formula))


def ask_human(question: str) -> str:
    """Ask the user a question"""
    print("----- Human Input Required -----")
    user_input = input(question + " ")
    return user_input or ""


# All functions that can be called by the LLM Agent
name_to_function_map: dict[str, Callable] = {
    get_current_location.__name__: get_current_location,
    calculate.__name__: calculate,
    finish.__name__: finish,
    supatool_search.__name__: supatool_search,
    supatool_execute.__name__: supatool_execute,
    ask_human.__name__: ask_human
}

supatool_name_to_cuid_map: dict[str, str] = {}

std_lib = [
    # {
    #     "function": {
    #         "name": "get_current_location",
    #         "description": "Get the current location of the user.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {},
    #             "required": []
    #         }
    #     },
    #     "type": "function"
    # },
    # {
    #     "function": {
    #         "name": "get_current_weather",
    #         "description": "Get the current weather in a given location.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "latitude": {
    #                     "type": "number"
    #                 },
    #                 "longitude": {
    #                     "type": "number"
    #                 },
    #                 "temperature_unit": {
    #                     "type": "string",
    #                     "enum": [
    #                         "celsius",
    #                         "fahrenheit"
    #                     ]
    #                 }
    #             },
    #             "required": [
    #                 "latitude",
    #                 "longitude",
    #                 "temperature_unit"
    #             ]
    #         }
    #     },
    #     "type": "function"
    # },
    {
        "function": {
            "name": "calculate",
            "description": "Calculate the result of a given formula.",
            "parameters": {
                "type": "object",
                "properties": {
                    "formula": {
                        "type": "string",
                        "description": "Numerical expression to compute the result of, in Python syntax."
                    }
                },
                "required": [
                    "formula"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "finish",
            "description": "Answer the user's question, and finish the conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Answer to the user's question."
                    }
                },
                "required": [
                    "answer"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "ask_human",
            "description": "Ask the user for additional context or information if you need it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "A question to ask the user."
                    }
                },
                "required": [
                    "question"
                ]
            }
        },
        "type": "function"
    },
    {
        "type": "function",
        "function": {
            "name": "supatool_search",
            "description": "Search the Supatool registry for more tools that can extend your functionality. If you do not have the ability to execute a task, search Supatool for more options.",
            "parameters": {
                "type": "object",
                "properties": {
                    "searchQuery": {
                        "type": "string",
                        "description": "A query to search for more tools that can execute actions. This should be in the format of a full sentence length question."
                    }
                },
                "required": ["searchQuery"],
                "additionalProperties": False
            }
        }
    }
]

dynamic_toolbox = std_lib.copy()

# QUESTION_PROMPT = "\
#  find the weather here then calcuate the JoshScore by doing a specific math calculation"
QUESTION_PROMPT = "\
 book me a reservation at Mestizo Modern Mexican in boston for august 24th 2024 at noon for"

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant who can answer multistep questions by sequentially calling functions. Follow a pattern of THOUGHT (reason step-by-step about which function to call next), ACTION (call a function to as a next step towards the final answer), OBSERVATION (output of the function). Reason step by step which actions to take to get to the answer. Follow the questions and tasks closely to best make the user satisfied. If you are unable to do something yourself, use Supatool to find tools that can extend your functionality.",
    },
    {
        "role": "user",
        "content": QUESTION_PROMPT,
    },
]


def run(messages: list[dict]) -> list[dict]:
    """
    Run the Agentic ReAct loop
    """
    max_iterations = 20
    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=dynamic_toolbox,
            tool_choice="required",
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        tool_calls = response_message.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                if function_name.endswith("supatool"):
                    function_to_call: Callable = name_to_function_map[supatool_execute.__name__]
                elif function_name not in name_to_function_map:
                    print(
                        f"Invalid function name: {function_name}")
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"Invalid function name: {function_name!r}",
                        }
                    )
                    continue
                else:
                    function_to_call: Callable = name_to_function_map[function_name]

                print("----- Selected Tool:", function_name, "-----")
                try:
                    function_args_dict = json.loads(
                        tool_call.function.arguments)
                except json.JSONDecodeError as exc:
                    # JSON decoding failed
                    print(f"Error decoding function arguments: {exc}")
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error decoding function call `{function_name}` arguments {tool_call.function.arguments!r}! Error: {exc!s}",
                        }
                    )
                    continue
                try:
                    print("----- Executing Tool with input params: ----")
                    print(json.dumps(function_args_dict))

                    if function_name.endswith("supatool"):
                        cuid = supatool_name_to_cuid_map[function_name]
                        function_response = function_to_call(
                            cuid, function_args_dict)
                    else:
                        function_response = function_to_call(
                            **function_args_dict)

                    print("----- Function Response: -----")
                    print(function_response)
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                    )
                except StopException as exc:
                    # Agent wants to stop the conversation (Expected)
                    print("----- TASK FINISHED -----")
                    print(f"Final message: '{exc!s}'")
                    return messages
                except Exception as exc:
                    # Unexpected error calling function
                    print(
                        f"Error calling function `{function_name}`: {type(exc).__name__}: {exc!s}"
                    )
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error calling function `{function_name}`: {type(exc).__name__}: {exc!s}!",
                        }
                    )
                    continue

    return messages


messages = run(messages)
# for message in messages:
#     if not isinstance(message, dict):
#         message = message.model_dump()
#     print(json.dumps(message, indent=2))
