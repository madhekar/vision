import mcp.types as types
from mcp.server import Server

# Initialize MCP server
app = Server("calculator-server")

class AdvancedMath:
    def __init__(self, multiplier: float = 1.0):
        self.multiplier = multiplier

    def multiply(self, a: float, b: float) -> float:
        return a * b * self.multiplier

math_instance = AdvancedMath(multiplier=1.5)

# Register tool with OpenClaw's MCP
@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="advanced_multiply",
            description="Multiplies two numbers and applies a predefined multiplier",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "type": "The first number"},
                    "b": {"type": "number", "type": "The second number"},
                },
                "required": ["a", "b"],
            },
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "advanced_multiply":
        result = math_instance.multiply(arguments["a"], arguments["b"])
        return [types.TextContent(type="text", text=f"Result: {result}")]
    raise ValueError(f"Unknown tool: {name}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(app.run())