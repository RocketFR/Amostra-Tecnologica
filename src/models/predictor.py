import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def carregar_dados():
    """Carrega os dados do arquivo CSV"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        caminho_csv = os.path.join(base_dir, '..', 'data', 'historico_jogos.csv')
        
        df = pd.read_csv(caminho_csv)
        
        # Verifica colunas necessárias
        colunas_necessarias = ['Jogador_1', 'Jogador_2', 'personagem_1', 'personagem_2', 
                             'vitorias_1', 'vitorias_2', 'fase', 'vencedor']
        if not all(col in df.columns for col in colunas_necessarias):
            raise ValueError("Colunas faltando no arquivo CSV")
        
        df['vencedor_bin'] = (df['vencedor'] == df['Jogador_1']).astype(int)
        return df
    except Exception as e:
        raise Exception(f"Erro ao carregar dados: {str(e)}")

def preparar_dados(df):
    """Prepara os dados para treinamento"""
    try:
        le_personagem = LabelEncoder()
        le_fase = LabelEncoder()
        le_jogador = LabelEncoder()  # Add this line
        
        # Codifica personagens
        todos_personagens = pd.concat([df['personagem_1'], df['personagem_2']]).unique()
        le_personagem.fit(todos_personagens)
        
        # Codifica jogadores
        todos_jogadores = pd.concat([df['Jogador_1'], df['Jogador_2']]).unique()
        le_jogador.fit(todos_jogadores)  # Add this block
        
        df['personagem_1_enc'] = le_personagem.transform(df['personagem_1'])
        df['personagem_2_enc'] = le_personagem.transform(df['personagem_2'])
        df['fase_enc'] = le_fase.fit_transform(df['fase'])
        df['jogador_1_enc'] = le_jogador.transform(df['Jogador_1'])  # Add this line
        df['jogador_2_enc'] = le_jogador.transform(df['Jogador_2'])  # Add this line
        
        X = df[['jogador_1_enc', 'jogador_2_enc', 'personagem_1_enc', 'personagem_2_enc', 'vitorias_1', 'vitorias_2', 'fase_enc']]
        y = df['vencedor_bin']
        
        return X, y, le_personagem, le_fase, le_jogador  # Return 5 values now
    except Exception as e:
        raise Exception(f"Erro ao preparar dados: {str(e)}")

def treinar_modelo(df=None):
    try:
        if df is None:
            df = carregar_dados()
        
        X, y, le_personagem, le_fase, le_jogador = preparar_dados(df)  # Recebe o novo encoder
        
        modelo = RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
        modelo.fit(X, y)
        
        return modelo, (le_personagem, le_fase, le_jogador)  # Inclui le_jogador nos encoders
    except Exception as e:
        raise Exception(f"Erro ao treinar modelo: {str(e)}")
        
def prever_resultado(modelo, encoders, novo_jogo):
    try:
        le_personagem, le_fase, le_jogador = encoders  # Desempacota o novo encoder
        
        # Verifica campos necessários (incluindo nomes)
        campos_necessarios = ['Jogador_1', 'Jogador_2', 'personagem_1', 'personagem_2', 'vitorias_1', 'vitorias_2', 'fase']
        if not all(campo in novo_jogo for campo in campos_necessarios):
            raise ValueError("Dados incompletos no novo jogo")
        
        # Prepara os dados (incluindo codificação dos nomes)
        dados = pd.DataFrame([novo_jogo])
        dados['personagem_1_enc'] = le_personagem.transform(dados['personagem_1'])
        dados['personagem_2_enc'] = le_personagem.transform(dados['personagem_2'])
        dados['fase_enc'] = le_fase.transform(dados['fase'])
        dados['jogador_1_enc'] = le_jogador.transform(dados['Jogador_1'])  # Novo
        dados['jogador_2_enc'] = le_jogador.transform(dados['Jogador_2'])  # Novo
        
        X = dados[['jogador_1_enc', 'jogador_2_enc', 'personagem_1_enc', 'personagem_2_enc', 'vitorias_1', 'vitorias_2', 'fase_enc']]
        
        predicao = modelo.predict(X)[0]
        return '1' if predicao == 1 else '2'
    except Exception as e:
        raise Exception(f"Erro ao fazer previsão: {str(e)}")

def selecionar_jogadores_aleatorios(df):
    """Seleciona um jogo aleatório a partir do DataFrame"""
    try:
        jogo = df.sample(n=1).iloc[0]  # Pega uma linha aleatória
        
        return {
            'Jogador_1': jogo['Jogador_1'],
            'personagem_1': jogo['personagem_1'],
            'vitorias_1': jogo['vitorias_1'],
            'Jogador_2': jogo['Jogador_2'],
            'personagem_2': jogo['personagem_2'],
            'vitorias_2': jogo['vitorias_2'],
            'fase': jogo['fase']
        }
    except Exception as e:
        raise Exception(f"Erro ao selecionar jogo aleatório: {str(e)}")
    


def salvar_dados(df, caminho='dados/dados_historicos.csv'):
    """Salva o DataFrame de volta no arquivo CSV"""
    try:
        df.to_csv(caminho, index=False)
        return True
    except Exception as e:
        print(f"Erro ao salvar dados: {str(e)}")
        return False