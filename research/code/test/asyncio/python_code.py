import asyncio
import sys

async def run_with_timeout():
    # Start python inside a subprocess expecting input over stdin
    process = await asyncio.create_subprocess_exec(
        sys.executable, '-',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE
    )

    code = "import time; time.sleep(1); print('Hello from the child process!')"

    try:
        # Send code to stdin and wait up to 2 seconds for execution
        stdout, _ = await asyncio.wait_for(
            process.communicate(input=code.encode()), 
            timeout=2.0
        )
        print(stdout.decode())
    except asyncio.TimeoutError:
        print("Process timed out! Killing it...")
        process.kill()
        await process.wait()

asyncio.run(run_with_timeout())
