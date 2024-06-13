import streamlit as st
import pandas as pd
import joblib

# Load models
logistic_model = joblib.load('checkpoints/logistic-regression.joblib')
random_forest_model = joblib.load('checkpoints/random_forest_model.joblib')

# Load and preprocess the dataset
df = pd.read_csv('data/quest.csv', encoding='ISO-8859-1')
df = df.drop('Personalidade', axis=1)

# Define mappings
value_map = {
    -3: "Discordo Totalmente",
    -2: "Discordo",
    -1: "Parcialmente Discordo",
    0: "Neutro",
    1: "Parcialmente Concordo",
    2: "Concordo",
    3: "Concordo Totalmente"
}
mapa = {
    0: 'ESFJ', 1: 'ESFP', 2: 'ESTJ', 3: 'ESTP', 
    4: 'ENFJ', 5: 'ENFP', 6: 'ENTJ', 7: 'ENTP', 
    8: 'ISFJ', 9: 'ISFP', 10: 'ISTJ', 11: 'ISTP', 
    12: 'INFJ', 13: 'INFP', 14: 'INTJ', 15: 'INTP'
}

descricoes = {
    'ESFJ': 'As pessoas que têm personalidade do tipo ESFJ são sérias e práticas, comprometidas com suas responsabilidades e sensíveis às necessidades dos outros. Elas se esforçam para alcançar a harmonia e são generosas com seu tempo, esforços e emoções. Os ESFJs estão sempre ansiosos para agradar, tanto no trabalho quanto em casa. Valorizam a lealdade, a tradição e seguem um código moral estrito. Em geral, gostam de suas rotinas e conservam uma programação regular, se mantendo organizados e produtivos.',
    'ESFP': 'Frequentemente vistas como "artistas", as pessoas que têm a personalidade do tipo ESFP são extrovertidas, amigáveis e generosas. Gostam de ficar perto de outras pessoas, espalhando entusiasmo e alegria em casa e no trabalho. Embora práticas e realistas em seu trabalho, elas também valorizam a diversão ao atingir seus objetivos. São enérgicas e flexíveis, incentivando os outros ao longo do caminho.',
    'ESTJ': 'Em geral, os ESTJs são grandes executivos, que valorizam a tradição e ordem. Traços fortes de caráter são importantes para ESTJs, que respeitam a honestidade e a dedicação, tanto consigo mesmo quanto com outras pessoas. Os ESTJs são tomadores de decisão práticos, procurando maneiras de obter resultados de forma rápida e eficaz. São organizados, lógicos e competentes tanto na criação quanto na implementação de projetos. Eles não se esquivam de planos ou decisões difíceis e trabalham para reunir outras pessoas em direção a um propósito comum.',
    'ESTP': 'O tipo de personalidade ESTP é frequentemente chamado de "empreendedor". As pessoas desse grupo são enérgicas, pragmáticas e flexíveis. Buscam avidamente por resultados rápidos, muitas vezes correndo riscos para encontrar as melhores soluções. Agir antes de pensar é algo comum para os ESTPs , que depois corrigem e adaptam seus processos conforme avançam. Os ESTPs desfrutam de um estilo de vida acelerado, vivendo o “momento presente” e passando bastante tempo perto de grupos de pessoas.',
    'ENFJ': 'A pessoa que tem o tipo de personalidade ENFJ costuma ser uma líder carismática e empática. Os ENFJs são altamente intuitivos quando se trata das emoções, necessidades e motivações dos outros. Com frequência, são leais e responsáveis, procurando maneiras de melhorar sua equipe, liderando com inspiração e receptividade. Também procuram oportunidades para tornar o mundo melhor e reunir pessoas que façam a diferença.',
    'ENFP': 'A pessoa com o tipo de personalidade ENFP pode ser poderosa em muitos locais de trabalho, pois geralmente é inovadora, inspiradora e não tem medo de correr riscos. Os ENFPs representam aproximadamente 8% da população e incluem mais mulheres do que homens. São pessoas muito perceptivas e que têm facilidade para compreender como os indivíduos e grupos funcionam, tornando-se líderes naturais dentro das organizações. Os ENFPs são entusiasmados, gostam de aprendizado abstrato e experiencial, e buscam o máximo potencial em suas experiências profissionais e pessoais.',
    'ENTJ': 'A pessoa que tem o tipo de personalidade ENTJ é comumente chamada de “comandante”. Os ENTJs são líderes naturais e costumam ser honestos, racionais e têm facilidade para tomar decisões. São rápidos em detectar ineficiências e desenvolver maneiras de resolver problemas. Os ENTJs valorizam muito o estabelecimento de metas, a organização e o planejamento. Carismáticos e confiantes, são eficazes para reunir um grupo em torno de um objetivo comum.',
    'ENTP': 'A pessoa com a personalidade do tipo ENTP tem forte pensamento empreendedor, quer decida por abrir seu próprio negócio ou inovar dentro de uma organização. Os ENTPs costumam ter dificuldade para cumprir prazos e não gostam de trabalhar dentro de hierarquias, o que pode restringir seus talentos inovadores. Preferem focar em uma “grande ideia” e resistir a tarefas rotineiras e repetitivas. Em vez disso, são mais atraídos por um trabalho conceitual e pela resolução de problemas, deixando os detalhes para os outros.',
    'ISFJ': 'Convencionais e tradicionais, os ISFJs respeitam e se esforçam para manter as estruturas estabelecidas, criar e manter ambientes ordenados. Eles têm uma forte ética de trabalho, que inclui servir aos outros, e são dedicados a seus deveres. São trabalhadores responsáveis e metódicos, que não ficam satisfeitos até que a tarefa seja totalmente concluída.',
    'ISFP': 'Normalmente, o tipo de personalidade ISFP é amigável e silencioso. São pessoas que gostam de observar e absorver o ambiente ao seu redor. De maneira geral, o ISFP prefere autonomia, trabalhando em seu próprio espaço e concluindo tarefas dentro de sua própria programação. Os ISFPs valorizam muito a lealdade e o comprometimento em seus relacionamentos pessoais. A harmonia também é importante para o ISFP, que tenta evitar confrontos, mantendo suas opiniões para si mesmo.',
    'ISTJ': 'Em geral, quem tem o tipo de personalidade ISTJ é prático e responsável. Essas pessoas tomam decisões lógicas e realizam tarefas de maneira estruturada e organizada. Os ISTJs costumam desfrutar de um espaço limpo e organizado, tanto em casa quanto no trabalho. São pessoas que valorizam muito as tradições, a lealdade, a rotina e a ordem, o que, por vezes, resulta na dificuldade de serem flexíveis em tempos de mudança.',
    'ISTP': 'Quem pertence ao grupo ISTP costuma ser quieto e observador. Quando surge um problema, essas pessoas são tolerantes, flexíveis e rápidas para encontrar uma solução. Organizado e prático, o ISTP valoriza dados, lógica e fatos para entender as questões. Geralmente, encontram significado no trabalho ao desenvolver e criar coisas, pensando em maneiras de fazer as coisas funcionarem e aprendendo ao longo do caminho.',
    'INFJ': 'Esta é a mais rara das 16 personalidades. O tipo INFJ é bastante perspicaz sobre as necessidades, motivações e preocupações das pessoas e costumam encontrar valor nos relacionamentos com os outros. Embora muitas vezes artísticos, criativos e complexos, os INFJs também são profundamente carinhosos e gentis. Muitas vezes chamadas de "defensoras", as pessoas com essa personalidade costumam encontrar um significado em trabalhos em que ajudam os outros.',
    'INFP': 'Como os INFPs são indivíduos bastante curiosos, inquisitivos e inovadores, geralmente são otimistas em sua visão de mundo e podem ser membros inspiradores para compor uma equipe. Os INFPs representam apenas 2 por cento da população. Eles são muito criativos, têm facilidade para encontrar conexões em padrões ocultos e gostam de pensamentos abstratos.',
    'INTJ': 'O tipo INTJ é guiado pela razão e pela lógica e está sempre em busca de adquirir e usar conhecimento. São pessoas muito seguras, que tentam reformar e melhorar o mundo ao seu redor. Embora autoconfiantes, os INTJs podem se sentir desconfortáveis em grandes grupos ou entre pessoas que não conhecem bem. preferem discutir ideias e fatos, ao invés de se envolver em conversas superficiais.',
    'INTP': 'As pessoas com o tipo de personalidade INTP tendem a ser quietas e contidas. Gostam de ideias abstratas e pensamentos profundos sobre teorias e a interação com os outros. Os INTPs são muito criativos, inteligentes, atenciosos e procuram respostas lógicas para as perguntas que surgem em seu ambiente. Em geral, os INTPs são céticos, analíticos e ótimos solucionadores de problemas. Essas pessoas são particularmente úteis quando surgem certas dificuldades no local de trabalho.'
}
st.title("Projeto Final - Trainees II")
st.markdown("Projeto final da diretoria de Trainees II, da TAIL. Nessa pesquisa, você deve responder o grau que você identifica com as seguintes afirmações, sendo -3 algo que você discorda totalmente que te define, e 3 algo que você concorda muito que está de acordo com você.")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

responses = []
for column in df.columns:
    response = st.slider(
        f"{column}",
        min_value=-3,
        max_value=3,
        value=0,  
        format="%d"
    )
    responses.append(response)

if st.button('Enviar'):
    new_row = pd.Series(responses, index=df.columns)
    new_df = pd.DataFrame([new_row])
    
    new_df.to_csv('data/nova_linha.csv', index=False)
    
    # Predict using both models
    logistic_prediction = logistic_model.predict(new_df)[0]
    random_forest_prediction = random_forest_model.predict(new_df)[0]
    
    # Get personality names and descriptions
    logistic_class_name = mapa[logistic_prediction]
    logistic_class_description = descricoes[logistic_class_name]
    
    random_forest_class_name = mapa[random_forest_prediction]
    random_forest_class_description = descricoes[random_forest_class_name]
    
    # Display results
    st.write(f'Sua personalidade se enquadra em (Logistic Regression): {logistic_class_name}')
    st.write(logistic_class_description)
    
    st.write(f'Sua personalidade se enquadra em (Random Forest): {random_forest_class_name}')
    st.write(random_forest_class_description)