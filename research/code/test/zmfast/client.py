import asyncio
from fastmcp import Client

async def main():

    client = Client("./server.py")

    async with client:
        tools = await client.list_tools()
        print("tools available")

        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

        print("\n")    

        response = await client.call_tool("greet")
        print(response.data)

        print("\n")


        response = await client.call_tool(
            "add",
            {"a": "33",
             "b": "87"}
        )
        print(response.data)

if __name__=="__main__":
    asyncio.run(main())    

