from selenium import webdriver
from selenium.webdriver.common.by import By
import csv
import time

navegador = webdriver.Chrome() # Chamar o Navegador

dados_produtos = [] # Guardar dados

total_paginas = 48 # Número de páginas

# Navegar pelas páginas
for pagina in range(1, total_paginas + 1):
    url = f'https://scrapeme.live/shop/page/{pagina}/'
    navegador.get(url)

    # Encontrar elementos de produto (nomes e preços)
    produtos = navegador.find_elements(By.CLASS_NAME, 'product')
    for produto in produtos:
        nome = produto.find_element(By.CLASS_NAME, 'woocommerce-loop-product__title').text
        preco = produto.find_element(By.CLASS_NAME, 'woocommerce-Price-amount').text
        imagem_url = produto.find_element(By.CSS_SELECTOR, 'img').get_attribute('src')  # Pega o URL da imagem
        dados_produtos.append([nome, preco, imagem_url])

with open('pokemons.csv', 'w', newline='', encoding='utf-8') as arquivo:
    escritor = csv.writer(arquivo)
    escritor.writerow(['Nome', 'Preço', 'Imagem'])
    escritor.writerows(dados_produtos)

print('Dados salvos em pokemons.csv')
