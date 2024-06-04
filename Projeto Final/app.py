import streamlit as st
import pandas as pd

# Carregar o dataset
# Substitua 'seu_arquivo.csv' pelo caminho do seu arquivo CSV
df = pd.read_csv('16P.csv', encoding='ISO-8859-1')


# Mapeamento de valores
value_map = {
    -3: "Discordo Totalmente",
    -2: "Discordo",
    -1: "Parcialmente Discordo",
    0: "Neutro",
    1: "Parcialmente Concordo",
    2: "Concordo",
    3: "Concordo Totalmente"
}

# Inverter o mapeamento para obter os números dos textos
inverse_value_map = {v: k for k, v in value_map.items()}

# Inicializar uma lista para armazenar as respostas
responses = []

# Criar o formulário
st.title("Questionário")

# Iterar sobre as colunas do dataset e criar uma pergunta para cada uma
for column in df.columns:
    response = st.radio(
        f"{column}",
        options=list(value_map.values()),
        index=3  # Índice inicial para "Neutro"
    )
    responses.append(inverse_value_map[response])

# Quando o formulário for submetido
if st.button('Enviar'):
    # Adicionar a nova linha de respostas ao dataframe
    new_row = pd.Series(responses, index=df.columns)
    df = df.append(new_row, ignore_index=True)
    
    # Salvar o dataframe atualizado de volta para o arquivo CSV (opcional)
    df.to_csv('seu_arquivo_atualizado.csv', index=False)
    
    st.success('Respostas enviadas com sucesso!')
    st.write(df)  # Mostrar o dataframe atualizado