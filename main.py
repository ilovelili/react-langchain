from typing import Union, List

from dotenv import load_dotenv
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool, tool
from langchain.tools.render import render_text_description


load_dotenv()

# This code snippet sets up a LangChain-based agent that uses a tool to answer questions.
# It defines a function to get the length of a given text and integrates it into a chain of operations that can be invoked to process input questions.


@tool
# This defines a tool function get_text_length that returns the length of a given text.
# It strips away certain characters before calculating the length.
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip('"')
    return len(text)


# This utility function searches for a tool by name in a list of tools and returns it.
# If the tool is not found, it raises a ValueError.
def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")


if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    tools = [get_text_length]

    # defines a prompt template for the agent, specifying the format for processing questions and actions.
    # The PromptTemplate is partially filled with the descriptions of the available tools and their names.
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought:
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    # initializes a ChatOpenAI language model with specific parameters,
    # including setting the temperature to 0 (deterministic responses) and defining stop sequences.
    # The model_kwargs parameter allows for additional arguments to be passed to the model. In this case, it specifies the stop sequences:
    # The stop parameter is used to define sequences where the model should stop generating further text. Here, it is set to stop on either "\nObservation" or "Observation".
    # This means that when the model generates text, it will halt as soon as it encounters either of these sequences. This is useful for controlling the format of the generated responses, especially in structured outputs like agent interactions.
    llm = ChatOpenAI(
        temperature=0, model_kwargs={"stop": ["\nObservation", "Observation"]}
    )

    intermediate_steps = []

    # This creates an agent chain using LangChain's expression language. The chain consists of:
    # An input mapping
    # The prompt template
    # The language model
    # An output parser (ReActSingleInputOutputParser)
    agent = (
        {
            "input": lambda x: x["input"],
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    # This invokes the agent with an example input question ("What is the length of 'DOG' in characters?").
    # The agent's response (agent_step) is printed.
    # The response will either be an AgentAction (indicating an action to take) or an AgentFinish (indicating a final answer)
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of 'DOG' in characters?",
        }
    )
    print(agent_step)

    # If the agent's response is an AgentAction, it extracts the tool name and input from the response.
    # It finds the corresponding tool using find_tool_by_name, invokes the tool with the input, and prints the observation (result of the tool's action).
    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"{observation=}")
