import asyncio
from crawlee.crawlers import PlaywrightCrawler, PlaywrightCrawlingContext

async def main() -> None:
    # Initialize the crawler settings
    crawler = PlaywrightCrawler(
        max_requests_per_crawl=20,  # Limits scope for safety
        headless=True,               # Runs browser hidden in the background
    )

    # Define how to handle every page the crawler encounters
    @crawler.router.default_handler
    async def request_handler(context: PlaywrightCrawlingContext) -> None:
        context.log.info(f"Processing: {context.request.url}")
        
        # Pull data safely from the rendered DOM
        page_data = {
            'url': context.request.url,
            'title': await context.page.title(),
        }
        
        # Save to local JSON storage Automatically
        await context.push_data(page_data)
        
        # Automatically find all internal links on the page and add them to the queue
        await context.enqueue_links()

    # Give it the initial start URL
    await crawler.run(['https://ycombinator.com'])

if __name__ == "__main__":
    asyncio.run(main())
