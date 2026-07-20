import sys
from fastmcp import FastMCP

'''
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}

{
  "jsonrpc": "2.0",
  "id":1,
  "method": "tools/call",
  "params": {
  "name": "add",
  "arguments": {
   "a" : "33",
   "b" : "4"
   }
  }
}
'''

mcp = FastMCP("zmedia fast MCP")

@mcp.tool()
def add(a: int, b: int) -> int:
    '''add two numbers'''
    return a + b

@mcp.tool()
def greet()->str:
    '''greet user to fast mcp server'''
    return "welcome to fast mcp!"

if __name__ == "__main__":
    mcp.run(transport="stdio")
