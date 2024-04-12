import csv
import asyncio
from playwright.async_api import Playwright, async_playwright

async def scrape_data(page):
    scraped_elements = []
    items = await page.query_selector_all("li.product")

    for i in items:
        scraped_element = {}

        el_title = await i.query_selector("h2")
        scraped_element["product"] = await el_title.inner_text()

        el_price = await i.query_selector("span.woocommerce-Price-amount")
        scraped_element["price"] = await el_price.text_content()

        image = await i.query_selector("a.woocommerce-LoopProduct-link.woocommerce-loop-product__link > img")
        scraped_element["img_link"] = await image.get_attribute("src")
        scraped_elements.append(scraped_element)
    return scraped_elements

async def run(playwright: Playwright, total_pages: int) -> None:
    browser = await playwright.chromium.launch(headless=False)
    context = await browser.new_context()
    
    all_data = []
    for page_number in range(1, total_pages + 1):
        page = await context.new_page()
        url = f"https://scrapeme.live/shop/page/{page_number}/"
        await page.goto(url)
        data = await scrape_data(page)
        print(f"Data from page {page_number}: {data}")
        all_data.extend(data)
        await page.close()
    
    await context.close()
    await browser.close()
    return all_data


async def main() -> None:
    total_pages = 48
    async with async_playwright() as playwright:
        data = await run(playwright, total_pages)
        save_as_csv(data)


def save_as_csv(data):
    with open("scraped_data.csv", "w", newline="",) as csvfile:
        fields = ["product", "price", "img_link"]
        writer = csv.DictWriter(csvfile, fieldnames=fields, quoting=csv.QUOTE_ALL, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)
        

asyncio.run(main())



