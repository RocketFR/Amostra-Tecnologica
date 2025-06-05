import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models.predictor import carregar_dados, treinar_modelo, prever_resultado, salvar_dados
from main import selecionar_jogadores_aleatorios

# Configuração da página
st.set_page_config(page_title="Preditor de Jogos", layout="wide")
st.title("🔮 Sistema de Predição de Vencedores")

# Carrega os dados
df = carregar_dados()

# Treina o modelo uma vez
modelo, encoders = treinar_modelo(df)
le_personagem, le_fase, le_jogador = encoders

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
        dados['jogador_1_enc'] = le_jogador.transform(dados['Jogador_1'])  # Added player encoding
        dados['jogador_2_enc'] = le_jogador.transform(dados['Jogador_2'])  # Added player encoding
        
        # Features must match exactly what was used during training
        X = dados[['jogador_1_enc', 'jogador_2_enc', 'personagem_1_enc', 'personagem_2_enc', 'vitorias_1', 'vitorias_2', 'fase_enc']]
        
        # Obter probabilidades
        proba = modelo.predict_proba(X)[0]
        
        # Verificar a ordem das classes no modelo
        if modelo.classes_[0] != 1:  # Se a ordem das classes estiver invertida
            proba = proba[::-1]  # Inverte as probabilidades
            
        # Definir cores baseado no vencedor
        cores = ['#4CAF50', '#F44336'] if st.session_state['resultado'] == '1' else ['#F44336', '#4CAF50']
        
        fig, ax = plt.subplots()
        bars = ax.bar([f'Jogador 1 ({jogador1})', f'Jogador 2 ({jogador2})'], proba, color=cores)
        ax.set_ylabel('Probabilidade de Vitória')
        ax.set_title('Chances de Vitória')
        
        # Adicionar valores exatos nas barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        st.pyplot(fig)

with col2:
    st.header("Análise de Dados Históricos")
    
    if 'personagem1' in st.session_state and 'personagem2' in st.session_state:
        # Criar DataFrame separado para cada jogador
        df_jogador1 = df[df['personagem_1'] == st.session_state['personagem1']].copy()
        df_jogador1['tipo_jogador'] = 'Jogador 1'
        
        df_jogador2 = df[df['personagem_2'] == st.session_state['personagem2']].copy()
        df_jogador2['tipo_jogador'] = 'Jogador 2'
        
        # Combinar os dados mantendo a distinção
        df_combined = pd.concat([df_jogador1, df_jogador2])
        
        # Ajustar a coluna de vitórias para cada caso
        df_combined['resultado'] = df_combined.apply(
            lambda x: x['vencedor_bin'] if x['tipo_jogador'] == 'Jogador 1' else (1 - x['vencedor_bin']),
            axis=1
        )
        
        # Gráfico de desempenho separado
        st.subheader("Desempenho Individual por Personagem")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # Plot separado para cada jogador
        sns.countplot(
            data=df_combined,
            x='tipo_jogador',
            hue='resultado',
            ax=ax2,
            order=['Jogador 1', 'Jogador 2'],
            palette={1: '#4CAF50', 0: '#F44336'}  # 1=Vitórias, 0=Derrotas
        )
        
        ax2.set_title(f'Desempenho de {st.session_state["personagem1"]} (J1) vs {st.session_state["personagem2"]} (J2)')
        ax2.set_xlabel('Jogador')
        ax2.set_ylabel('Contagem')
        ax2.legend(['Derrotas', 'Vitórias'])
        
        st.pyplot(fig2)

# Seção de Edição de Dados Históricos
st.header("📝 Editor de Dados Históricos")

tab1, tab2, tab3 = st.tabs(["Visualizar Dados", "Editar Registros", "Adicionar Novo"])

with tab1:
    st.subheader("Visualização Completa do Histórico")
    st.dataframe(df, height=400)

with tab2:
    st.subheader("Editar Registro Existente")
    
    registros = df.to_dict('records')
    registro_selecionado = st.selectbox(
        "Selecione um registro para editar:",
        options=range(len(registros)),
        format_func=lambda x: f"Jogo {x+1}: {registros[x]['Jogador_1']} vs {registros[x]['Jogador_2']}"
    )
    
    with st.form("editar_form"):
        registro = registros[registro_selecionado]
        
        col1, col2 = st.columns(2)
        with col1:
            novo_jogador1 = st.text_input("Jogador 1", registro['Jogador_1'])
            novo_personagem1 = st.selectbox(
                "Personagem 1", 
                le_personagem.classes_,
                index=list(le_personagem.classes_).index(registro['personagem_1'])
            )
            novo_vitorias1 = st.number_input(
                "Vitórias 1", 
                min_value=0, 
                value=registro['vitorias_1']
            )
        
        with col2:
            novo_jogador2 = st.text_input("Jogador 2", registro['Jogador_2'])
            novo_personagem2 = st.selectbox(
                "Personagem 2", 
                le_personagem.classes_,
                index=list(le_personagem.classes_).index(registro['personagem_2'])
            )
            novo_vitorias2 = st.number_input(
                "Vitórias 2", 
                min_value=0, 
                value=registro['vitorias_2']
            )
        
        nova_fase = st.selectbox(
            "Fase", 
            le_fase.classes_,
            index=list(le_fase.classes_).index(registro['fase'])
        )
        
        novo_vencedor = st.selectbox(
            "Vencedor", 
            [1, 2],
            index=0 if registro.get('vencedor_bin', 1) == 1 else 1
        )
        
        if st.form_submit_button("Salvar Alterações"):
            # Atualizar o DataFrame
            df.at[registro_selecionado, 'Jogador_1'] = novo_jogador1
            df.at[registro_selecionado, 'Jogador_2'] = novo_jogador2
            df.at[registro_selecionado, 'personagem_1'] = novo_personagem1
            df.at[registro_selecionado, 'personagem_2'] = novo_personagem2
            df.at[registro_selecionado, 'vitorias_1'] = novo_vitorias1
            df.at[registro_selecionado, 'vitorias_2'] = novo_vitorias2
            df.at[registro_selecionado, 'fase'] = nova_fase
            df.at[registro_selecionado, 'vencedor_bin'] = novo_vencedor
            
            salvar_dados(df)
            st.success("Registro atualizado com sucesso!")
            st.experimental_rerun()

with tab3:
    st.subheader("Adicionar Novo Registro")
    
    with st.form("novo_form"):
        col1, col2 = st.columns(2)
        with col1:
            novo_jogador1 = st.text_input("Jogador 1", "Nome")
            novo_personagem1 = st.selectbox("Personagem 1", le_personagem.classes_)
            novo_vitorias1 = st.number_input("Vitórias 1", min_value=0, value=10)
        
        with col2:
            novo_jogador2 = st.text_input("Jogador 2", "Nome")
            novo_personagem2 = st.selectbox("Personagem 2", le_personagem.classes_)
            novo_vitorias2 = st.number_input("Vitórias 2", min_value=0, value=10)
        
        nova_fase = st.selectbox("Fase", le_fase.classes_)
        novo_vencedor = st.selectbox("Vencedor", [1, 2])
        
        if st.form_submit_button("Adicionar Registro"):
            novo_registro = {
                'Jogador_1': novo_jogador1,
                'Jogador_2': novo_jogador2,
                'personagem_1': novo_personagem1,
                'personagem_2': novo_personagem2,
                'vitorias_1': novo_vitorias1,
                'vitorias_2': novo_vitorias2,
                'fase': nova_fase,
                'vencedor_bin': novo_vencedor
            }
            
            df = pd.concat([df, pd.DataFrame([novo_registro])], ignore_index=True)
            salvar_dados(df)
            st.success("Novo registro adicionado com sucesso!")
            st.experimental_rerun()