
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def carregar_dados(caminho_csv='src/data/historico.csv'):
    df = pd.read_csv(caminho_csv)
    df['vencedor_bin'] = (df['vencedor'] == df['jogador_1']).astype(int)
    return df

def preparar_dados(df):
    le_personagem = LabelEncoder()
    le_fase = LabelEncoder()

    df['personagem_1'] = le_personagem.fit_transform(df['personagem_1'])
    df['personagem_2'] = le_personagem.transform(df['personagem_2'])
    df['fase'] = le_fase.fit_transform(df['fase'])

    X = df[['personagem_1', 'personagem_2', 'vitorias_1', 'vitorias_2', 'fase']]
    y = df['vencedor_bin']
    return X, y, le_personagem, le_fase
