import pandas as pd
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

SYSTEM_PROMPT = """You are an expert Python Pandas programmer and analyst. You answer questions
related to the provided dataset by running pandas code against a DataFrame called `df`.

IMPORTANT — follow these rules exactly:
1. To get data you MUST call the `execute_pandas_code` tool with valid Python pandas code.
2. The code you pass must be a one or more Python expression that operates on `df`.
3. Do NOT import anything — pandas is already available as `pd` and the DataFrame as `df`.
4. Do NOT use print(); just write the expression and its result will be returned.
5. If the tool returns an error, revise the code and try again (up to 3 times).
6. Use vectorized operations; avoid iterating over rows.
7. After receiving the result, summarise the answer in plain language.

Here are the columns in the DataFrame:
{columns}

Here are the first 5 rows:
{head}
"""


# create the dataframe
df = pd.read_csv("./Titanic-Dataset.csv")


SYSTEM_PROMPT = SYSTEM_PROMPT.format(
    columns=df.columns.tolist(),
    head=df.head().to_string()
)

df.head()

'''
The Tool

AI agents need tools to be able to perform their work. In terms of a python program, 
the simplest tool can be a function that will be called by the agent.
In the real production environment, there will be multiple tools that will be used by the agent, 
and they can be of different types (APIs, databases, python functions, etc..).


Let's create a simple function which will execute all pandas code passed to 
it and will be written by the LLM with the help of AI Agent
'''

@tool
def execute_pandas_code(code: str) -> str:
    '''tool be called by the agent to execute pandas code'''
    df_agent = df
    try:
        result = eval(code, {"__builtins__": {}}, {"df": df_agent, "pd": pd})
        return str(result)
    except Exception as e:
        return f"Error: {e}"
    

'''
tool ready, create the instance of LLM which will be used by agent. 
using Ollama here, but any LLM with tooling support can be used here
'''    

llm = ChatOllama(model="gpt-oss:latest", temperature=0.0)

agent_tools = [
            execute_pandas_code,
        ]

#create agent using Langchain
agent = create_agent(
            model=llm,
            tools=agent_tools,
            system_prompt=SYSTEM_PROMPT
        )

'''
Now its time to trigger our agent with a question related to titanic dataset
'''

question = "How many passengers survived in the titanic disaster"

# invoke the Agent

result = agent.invoke(
            {"messages": question}
        )

result["messages"][-1].content

'''
Streaming the answer from the agent

The agent can also stream the answer back to us and can show the conversation.
'''

for next_step in agent.stream(
            {"messages": question},
            stream_mode="values",
    ):
        next_step["messages"][-1].pretty_print()