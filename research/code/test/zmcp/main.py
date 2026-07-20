from fastmcp import FastMCP

zm = FastMCP("Zmedia MCP")

@zm.tool()
def add_num(a: int, b: int) -> int:
    return a + b 

@zm.tool()
def greet_user(name: str) -> str:
    print(f"Hello from zmcp {name}!")


if __name__ == "__main__":
    zm.run()
