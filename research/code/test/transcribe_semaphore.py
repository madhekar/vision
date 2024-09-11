import os
import asyncio
import aiohttp
import aiofiles
from typing import List


async def async_transcribe_audio(
    session: aiohttp.ClientSession,
    input_file_path: str,
    output_file_path: str,
    rate_limit_semaphore: asyncio.Semaphore,
):
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {os.environ.get("OPENAI_API_KEY")}"}

    async with rate_limit_semaphore:
        print(f"Starting transcription for "{input_file_path}"")
        async with aiofiles.open(input_file_path, "rb") as file:
            data = aiohttp.FormData()
            data.add_field(
                "file",
                file,
                filename=input_file_path.split("/")[-1],
                content_type="audio/mpeg",
            )
            data.add_field("model", "whisper-1")
            data.add_field("language", "en")
            data.add_field("prompt", "Welcome to our radio show.")
            data.add_field("response_format", "json")
            data.add_field("temperature", "0.2")

            try:
                async with session.post(url, headers=headers, data=data) as response:
                    response.raise_for_status()
                    transcription = await response.json()
                    
                    remaining_requests = int(
                        response.headers.get("x-ratelimit-remaining-requests", 0)
                    )
                    total_requests = int(
                        response.headers.get("x-ratelimit-limit-requests", 1)
                    )
                    remaining_percentage = remaining_requests / total_requests
                    print(f"Rate: {remaining_requests}/{total_requests}: {remaining_percentage*100:.1f}%")

                    if remaining_percentage < 0.1:  # Adjust this threshold as needed
                        backoff = 5  # Add a 5-second pause
                        print(
                            f"Low remaining requests ({remaining_requests}/{total_requests}). Retrying in {backoff} seconds."
                        )
                        await asyncio.sleep(backoff)
            except Exception as e:
                if response.status == 429:
                    if "check your plan" in response.text:
                        print("Aborting due to exceeded quota.")
                        return

                        return await async_transcribe_audio(
                            session,
                            input_file_path,
                            output_file_path,
                            rate_limit_semaphore,
                        )
                print(f"An API error occurred: {e}")
                # do some more backoff, retry, or handing cases
                return

        print(f"Sending complete for "{input_file_path}"")

    transcribed_text = transcription["text"]

    try:
        async with aiofiles.open(output_file_path, "w") as file:
            await file.write(transcribed_text)
        print(f"--- Transcribed text successfully saved to "{output_file_path}".")
    except Exception as e:
        print(f"Output file error: {e}")


async def main(input_file_paths: List[str]):
    rate_limit_semaphore = asyncio.Semaphore(
        20
    )  # Allow at least one request every 3 seconds
    output_file_paths = [path + "-transcript.txt" for path in input_file_paths]

    async with aiohttp.ClientSession() as session:
        tasks = [
            async_transcribe_audio(
                session, input_file_path, output_file_path, rate_limit_semaphore
            )
            for input_file_path, output_file_path in zip(
                input_file_paths, output_file_paths
            )
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    input_file_paths = ["audio1.mp3","audio2.mp3","audio3.mp3",]
    asyncio.run(main(input_file_paths))