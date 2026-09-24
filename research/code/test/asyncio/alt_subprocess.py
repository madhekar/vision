import asyncio

async def run_command():
    proc = await asyncio.create_subprocess_exec(
        'ls', '-l',
        stdout=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    print(stdout.decode())

asyncio.run(run_command())
