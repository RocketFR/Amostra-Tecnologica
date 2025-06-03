import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models.predictor import carregar_dados, treinar_modelo, prever_resultado
from main import selecionar_jogadores_aleatorios

# Configuração da página
st.set_page_config(page_title="Preditor de Jogos", layout="wide")
st.title("🔮 Sistema de Predição de Vencedores")

# Carrega os dados
df = carregar_dados()

# Treina o modelo uma vez
modelo, encoders = treinar_modelo(df)
le_personagem, le_fase = encoders

# Sidebar para entrada de dados
with st.sidebar:
    st.header("Configuração do Jogo")
    
    if st.button("Selecionar Jogadores Aleatorios No Historico"):
        jogo_aleatorio = selecionar_jogadores_aleatorios(df)
        st.session_state['jogador1'] = jogo_aleatorio['Jogador_1']
        st.session_state['personagem1'] = jogo_aleatorio['personagem_1']
        st.session_state['vitorias1'] = jogo_aleatorio['vitorias_1']
        st.session_state['jogador2'] = jogo_aleatorio['Jogador_2']
        st.session_state['personagem2'] = jogo_aleatorio['personagem_2']
        st.session_state['vitorias2'] = jogo_aleatorio['vitorias_2']
        st.session_state['fase'] = jogo_aleatorio['fase']
    
    jogador1 = st.text_input("Jogador 1", st.session_state.get('jogador1', 'Lucas'))
    personagem1 = st.selectbox("Personagem 1", le_personagem.classes_, index=le_personagem.classes_.tolist().index(st.session_state.get('personagem1', le_personagem.classes_[0])))
    vitorias1 = st.number_input("Vitórias 1", min_value=0, value=st.session_state.get('vitorias1', 15))
    
    jogador2 = st.text_input("Jogador 2", st.session_state.get('jogador2', 'Bruno'))
    personagem2 = st.selectbox("Personagem 2", le_personagem.classes_, index=le_personagem.classes_.tolist().index(st.session_state.get('personagem2', le_personagem.classes_[0])))
    vitorias2 = st.number_input("Vitórias 2", min_value=0, value=st.session_state.get('vitorias2', 7))
    
    fase = st.selectbox("Fase", le_fase.classes_, index=le_fase.classes_.tolist().index(st.session_state.get('fase', le_fase.classes_[0])))
    
    if st.button("Prever Vencedor"):
        novo_jogo = {
            'Jogador_1': jogador1,
            'Jogador_2': jogador2,
            'personagem_1': personagem1,
            'personagem_2': personagem2,
            'vitorias_1': vitorias1,
            'vitorias_2': vitorias2,
            'fase': fase
        }
        
        palpite = prever_resultado(modelo, encoders, novo_jogo)
        st.session_state['resultado'] = palpite
        st.session_state['jogo'] = novo_jogo
        st.session_state['personagem1'] = personagem1
        st.session_state['personagem2'] = personagem2
        st.session_state['fase'] = fase

# Área principal
col1, col2 = st.columns(2)

with col1:
    if 'resultado' in st.session_state:
        st.header("Resultado da Predição")
        vencedor = st.session_state['jogo'][f'Jogador_{st.session_state["resultado"]}']
        perdedor = st.session_state['jogo'][f'Jogador_{1 if st.session_state["resultado"] == "2" else 2}']
        
        st.success(f"🏆 {vencedor} tem maior chance de vencer contra {perdedor}!")
        
        # Gráfico de probabilidades
        dados = pd.DataFrame([st.session_state['jogo']])
        dados['personagem_1_enc'] = le_personagem.transform(dados['personagem_1'])
        dados['personagem_2_enc'] = le_personagem.transform(dados['personagem_2'])
        dados['fase_enc'] = le_fase.transform(dados['fase'])
        
        X = dados[['personagem_1_enc', 'personagem_2_enc', 'vitorias_1', 'vitorias_2', 'fase_enc']]
        proba = modelo.predict_proba(X)[0]
        
        fig, ax = plt.subplots()
        ax.bar([f'Jogador 1 ({jogador1})', f'Jogador 2 ({jogador2})'], proba, color=['#4CAF50', '#F44336'])
        ax.set_ylabel('Probabilidade de Vitória')
        ax.set_title('Chances de Vitória')
        st.pyplot(fig)

with col2:
    st.header("Análise de Dados Históricos")
    
    # Filtra os dados históricos para os personagens selecionados
    if 'personagem1' in st.session_state and 'personagem2' in st.session_state:
        personagens_selecionados = [st.session_state['personagem1'], st.session_state['personagem2']]
        df_filtrado = df[df['personagem_1'].isin(personagens_selecionados) | df['personagem_2'].isin(personagens_selecionados)]
        
        # Gráfico de distribuição de vitórias por personagem
        st.subheader("Vitórias por Personagem (Filtrado)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df_filtrado, x='personagem_1', hue='vencedor_bin', ax=ax2)
        ax2.set_title('Distribuição de Vitórias por Personagem')
        ax2.legend(['Perdeu', 'Venceu'])
        st.pyplot(fig2)
        
        # Gráfico de vitórias por fase (filtrado)
        if 'fase' in st.session_state:
            st.subheader(f"Vitórias na Fase: {st.session_state['fase']}")
            df_fase = df_filtrado[df_filtrado['fase'] == st.session_state['fase']]
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.countplot(data=df_fase, x='personagem_1', hue='vencedor_bin', ax=ax3)
            ax3.set_title(f'Vitórias na Fase {st.session_state["fase"]}')
            ax3.legend(['Perdeu', 'Venceu'])
            st.pyplot(fig3)

# Rodar com: python -m streamlit run src/app.py
# pip install matplotlib pandas scikit-learn streamlit seaborn