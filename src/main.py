from models.predictor import treinar_modelo, prever_resultado, carregar_dados
import random
import pandas as pd

def selecionar_jogadores_aleatorios(df):
    """Seleciona dois jogadores aleatórios com estatísticas realistas"""
    try:
        # Correção: Forma correta de concatenar as colunas
        todos_jogadores = pd.concat([df['Jogador_1'], df['Jogador_2']]).unique().tolist()
        
        if len(todos_jogadores) < 2:
            raise ValueError("Não há jogadores suficientes no histórico")
        
        # Seleciona 2 jogadores diferentes
        jogador1, jogador2 = random.sample(todos_jogadores, 2)
        
        # Tenta encontrar um jogo entre eles
        mascara = ((df['Jogador_1'] == jogador1) & (df['Jogador_2'] == jogador2)) | \
                 ((df['Jogador_1'] == jogador2) & (df['Jogador_2'] == jogador1))
        jogos = df[mascara]
        
        if not jogos.empty:
            jogo = jogos.sample(1).iloc[0]
            return {
                'Jogador_1': jogo['Jogador_1'],
                'Jogador_2': jogo['Jogador_2'],
                'personagem_1': jogo['personagem_1'],
                'personagem_2': jogo['personagem_2'],
                'vitorias_1': jogo['vitorias_1'],
                'vitorias_2': jogo['vitorias_2'],
                'fase': jogo['fase']
            }
        
        # Se não encontrou jogo entre eles, cria um novo
        return {
            'Jogador_1': jogador1,
            'Jogador_2': jogador2,
            'personagem_1': random.choice(df['personagem_1'].unique()),
            'personagem_2': random.choice(df['personagem_2'].unique()),
            'vitorias_1': random.randint(1, 30),
            'vitorias_2': random.randint(1, 30),
            'fase': random.choice(df['fase'].unique())
        }
    except Exception as e:
        raise Exception(f"Erro ao selecionar jogadores: {str(e)}")

if __name__ == '__main__':
    try:
        print("🚀 Iniciando sistema de previsão de jogos...")
        
        # Carrega dados
        print("\n📂 Carregando dados históricos...")
        df = carregar_dados()
        
        # Seleciona jogadores
        print("\n🎲 Selecionando jogadores aleatórios...")
        novo_jogo = selecionar_jogadores_aleatorios(df)
        
        # Treina modelo
        print("\n🤖 Treinando modelo...")
        modelo, encoders = treinar_modelo(df)
        
        # Mostra detalhes
        print("\n📊 Detalhes do jogo:")
        print(f"• Jogador 1: {novo_jogo['Jogador_1']} ({novo_jogo['personagem_1']}) - 🏆 {novo_jogo['vitorias_1']} vitórias")
        print(f"• Jogador 2: {novo_jogo['Jogador_2']} ({novo_jogo['personagem_2']}) - 🏆 {novo_jogo['vitorias_2']} vitórias")
        print(f"• 🎮 Fase: {novo_jogo['fase']}")
        
        # Faz previsão
        palpite = prever_resultado(modelo, encoders, novo_jogo)
        print(f"\n🔮 Palpite da IA: Jogador {palpite} tem maior chance de vencer!")
        
    except Exception as e:
        print(f"\n❌ Erro durante a execução: {str(e)}")