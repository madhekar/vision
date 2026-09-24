import asyncio

async def run_git_version():
    # Start the subprocess and redirect standard output to a pipe
    process = await asyncio.create_subprocess_exec(
        'git', '--version',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # Wait for the process to finish and read all output
    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        print(f"Success: {stdout.decode().strip()}")
    else:
        print(f"Error: {stderr.decode().strip()}")

asyncio.run(run_git_version())
