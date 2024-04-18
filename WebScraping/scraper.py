# Import libraries to deploy into scraper 
import asyncio 
# import a csv package to output the cleaned data to a .csv file 
import csv

from playwright.async_api import Playwright, async_playwright 



# Optionally, you might want to store output data in a .csv format 
def save_as_csv(data): 
	with open("scraped_data.csv", "w", newline="") as csvfile: 
		fields = ["product", "price", "img_link"] 
		writer = csv.DictWriter(csvfile, fieldnames=fields, quoting=csv.QUOTE_ALL) 
		writer.writeheader() 
		writer.writerows(data)

 
# Start with playwright scraping here: 
async def scrape_data(page): 
	scraped_elements = [] 
	items = await page.query_selector_all("li.product") 
 
	# Pick the scraping item 
	for i in items: 
		scraped_element = {} 
 
		# Product name 
		el_title = await i.query_selector("h2") 
		scraped_element["product"] = await el_title.inner_text() 
 
		# Product price 
		el_price = await i.query_selector("span.woocommerce-Price-amount") 
		scraped_element["price"] = await el_price.text_content() 

		# Product image 
		image = await i.query_selector( 
			"a.woocommerce-LoopProduct-link.woocommerce-loop-product__link > img" 
		) 
		scraped_element["img_link"] = await image.get_attribute("src") 
		scraped_elements.append(scraped_element)

	return scraped_elements 
 
 
async def run(playwright: Playwright) -> None: 
	# Launch the headed browser instance (headless=False) 
	# To see the process of playwright scraping 
	# chromium.launch - opens a Chromium browser 
	browser = await playwright.chromium.launch(headless=False) 
 
	# Creates a new browser context 
	context = await browser.new_context() 
 
	# Open new page 
	page = await context.new_page() 
 
	# Go to the chosen website 
	await page.goto("https://scrapeme.live/shop/") 
	data = await scrape_data(page) 
	# Go through different pages 
	for i in range(47): #Como eu dou retrieve no número de páginas existentes?
		await page.locator("text=→").nth(1).click() 
		data.extend(await scrape_data(page)) 
		await page.wait_for_selector("li.product")

	save_as_csv(data) # Save the retried data to csv

	

 
	print(data) 
 
	await context.close() 
	# Turn off the browser once you finished 
	await browser.close() 
 
 
async def main() -> None: 
	async with async_playwright() as playwright: 
		await run(playwright) 
 
 
asyncio.run(main())
