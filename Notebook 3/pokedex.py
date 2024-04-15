import requests
import pandas as pd
from bs4 import BeautifulSoup

lista_pokemons = []
lista_preços = []

url_base = ('https://scrapeme.live/shop/page/')

for i in range(1, 49):
    i_str = str(i)
    response = requests.get(url_base + i_str)

    content = response.content

    # HTML da pokedex
    site = BeautifulSoup(content, "html.parser")

    # HTML Pokemons
    pokemons = site.findAll("h2", attrs={"class": "woocommerce-loop-product__title"})
    prices = site.findAll("span", attrs={"class": "price"})
    for pokemon in pokemons:
        lista_pokemons.append(pokemon.text)

    # HTML Prices
    prices = site.findAll("span", attrs={"class": "price"})
    for price in prices:
        lista_preços.append(price.text)

# Usando pandas para criar o nosso csv
pokedex = pd.DataFrame(lista_pokemons, columns=['Pokémon']) # criar df
pokedex["Price"] = lista_preços # adicionar mais uma coluna

pokedex.to_csv("pokedex.csv")

