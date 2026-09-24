import asyncio
import sys


class SubprocessRunner:

  async def run_command(self, program: str, *args: str) -> tuple[bytes, bytes, int]:
    # Create the subprocess and capture stdout and stderr
    process = await asyncio.create_subprocess_exec(
        program, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    # Await communicate() to collect stdout, stderr, and finish the process
    stdout, stderr = await process.communicate()
    return stdout, stderr, process.returncode


# Example usage
async def main():
  runner = SubprocessRunner()
  # Runs: python3 --version
  stdout, stderr, returncode = await runner.run_command(sys.executable, "--version")
  print(f"Return Code: {returncode}")
  print(f"Stdout: {stdout.decode().strip()}")
  print(f"Stderr: {stderr.decode().strip()}")


if __name__ == "__main__":
  asyncio.run(main())
