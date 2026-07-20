import sys
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server with a distinct name
mcp = FastMCP("Math-and-Greeting-Server")

@mcp.tool()
def add_numbers(a: float, b: float) -> float:
    """
    Add two numbers together.
    The docstring serves as instructions for the LLM on when to call this tool.
    """
    return a + b

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """
    Expose a dynamic, read-only text file containing a customized greeting.
    """
    return f"Hello, {name}! Welcome to your custom Python MCP Server."

if __name__ == "__main__":
    # Local hosts communicate via standard input/output (stdio)
    mcp.run(transport="stdio")
