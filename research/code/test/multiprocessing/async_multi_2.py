import asyncio
import csv
import time
from string import Template
from urllib.parse import urljoin

from aiohttp import request
from aiomultiprocess import Pool
from bs4 import BeautifulSoup

list_url_t = Template("https://books.toscrape.com/catalogue/category/books_1/page-$page.html")


def get_detail_url(base_url: str, html: str) -> list[str]:
    """ Grab the link to the detail page of each book from the HTML code of the list page """
    result = []
    soup = BeautifulSoup(html, "html.parser")
    a_tags = soup.select("article.product_pod div.image_container a")
    for a_tag in a_tags:
        result.append(urljoin(base_url, a_tag.get("href")))
    return result


def parse_detail_page(html):
    """ Parse the HTML of the detail page to get the desired book data """
    soup = BeautifulSoup(html, "lxml")
    title = soup.select_one("div.product_main h1").text
    price = soup.select_one("div.product_main p.price_color").text
    description_tag = soup.select_one("div#product_description + p")
    description = description_tag.text if description_tag else ""

    return {"title": title, "price": price, "description": description}


async def fetch_list(url: str) -> list[str]:
    """ Get the URL of each detail page from the list page URL """
    print(f"fetch_list: begin to process url: {url}")
    async with request("GET", url) as response:
        html = await response.text()
        urls = get_detail_url(url, html)
    return urls


async def fetch_detail(url: str) -> dict:
    """ Get the book data on the detail page from the detail page URL """
    async with request("GET", url) as response:
        html = await response.text()
        detail = parse_detail_page(html)
        return detail


def write_to_csv(all_books: list):
    """ Writing data to CSV files """
    print(f"write_to_csv: begin to write books detail to csv.")
    with open("./scraping_result.csv", "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = all_books[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerows(all_books)


# async def asyncio_main():
#     """ Implementing web scraping by using asyncio alone """
#     start = time.monotonic()
#     all_books, detail_urls = [], []
#     fetch_list_tasks = [asyncio.create_task(fetch_list(list_url_t.substitute(page=i + 1))) for i in range(5)]
#     for urls in asyncio.as_completed(fetch_list_tasks):
#         detail_urls.extend(await urls)

#     fetch_detail_tasks = [asyncio.create_task(fetch_detail(detail_url)) for detail_url in detail_urls]
#     for detail in asyncio.as_completed(fetch_detail_tasks):
#         all_books.append(await detail)
#     write_to_csv(all_books)
#     print(f"All done in {time.monotonic() - start} seconds")


# if __name__ == "__main__":
#     asyncio.run(asyncio_main())

async def aiomultiprocess_main():
    """
    Integrating multiprocessing and asyncio with the help of aiomultiprocess,
    requires only a simple rewriting of the main function
    """
    start = time.monotonic()
    all_books = []
    async with Pool() as pool:
        detail_urls = []
        async for urls in pool.map(fetch_list,
                                   [list_url_t.substitute(page=i + 1) for i in range(5)]):
            detail_urls.extend(urls)

        async for detail in pool.map(fetch_detail, detail_urls):
            all_books.append(detail)

    write_to_csv(all_books)
    print(f"All done in {time.monotonic() - start} seconds")


if __name__ == "__main__":
    asyncio.run(aiomultiprocess_main())