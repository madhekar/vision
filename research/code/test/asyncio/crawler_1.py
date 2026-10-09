import asyncio
# Correct sub-module path for all crawlers
from crawlee.crawlers import FileDownloadCrawler, FileDownloadCrawlingContext

async def main() -> None:
    # Initialize the binary file downloader
    crawler = FileDownloadCrawler()

    @crawler.router.default_handler
    async def request_handler(context: FileDownloadCrawlingContext) -> None:
        context.log.info(f'Successfully downloaded image from: {context.request.url}')
        # The raw bytes are stored in context.http_response.body
        # Crawlee automatically dumps these files into your key_value_stores directory

    # Run the crawler with an image URL
    await crawler.run([
        'https://amazon.com',
    ])

if __name__ == '__main__':
    asyncio.run(main())

