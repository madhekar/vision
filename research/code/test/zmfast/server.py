import sys
from fastmcp import FastMCP

'''
{
  "jsonrpc": "2.0",
  "id":1,
  "method": "add",
  "params": {
   "a" : "33",
   "b" : "4"
  },
  "clientInfo": {
  "name": "zmedia",
  "version": "4.5"
  }
}
'''

mcp = FastMCP("zmedia fast MCP")

@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    mcp.run(transport="stdio")
