
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def carregar_dados(caminho_csv='src/data/historico.csv'):
    df = pd.read_csv(caminho_csv)
    df['vencedor_bin'] = (df['vencedor'] == df['jogador_1']).astype(int)
    return df

def preparar_dados(df):
    # Inicializa os encoders
    le_personagem = LabelEncoder()
    le_fase = LabelEncoder()
    le_jogador = LabelEncoder()  # Novo encoder para jogadores

    # Codifica personagens e fase (como antes)
    df['personagem_1'] = le_personagem.fit_transform(df['personagem_1'])
    df['personagem_2'] = le_personagem.transform(df['personagem_2'])
    df['fase'] = le_fase.fit_transform(df['fase'])

    # Codifica os nomes dos jogadores (novo)
    todos_jogadores = pd.concat([df['Jogador_1'], df['Jogador_2']]).unique()
    le_jogador.fit(todos_jogadores)
    df['jogador_1_enc'] = le_jogador.transform(df['Jogador_1'])
    df['jogador_2_enc'] = le_jogador.transform(df['Jogador_2'])

    # Adiciona as novas features ao X
    X = df[['jogador_1_enc', 'jogador_2_enc', 'personagem_1', 'personagem_2', 'vitorias_1', 'vitorias_2', 'fase']]
    y = df['vencedor_bin']

    return X, y, le_personagem, le_fase, le_jogador  # Retorna o novo encoder