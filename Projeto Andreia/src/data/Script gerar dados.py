import pandas as pd
import random


jogadores = ["Lucas", "Bruno", "Mariana", "Rafa", "João"]
personagens = ["Mario", "Link", "Pikachu", "Ness", "Samus"]
fases = ["Final_Destination", "Battlefield", "Smashville", "Pokemon_Stadium"]


partidas = []
for _ in range(50):
    j1, j2 = random.sample(jogadores, 2)
    p1 = random.choice(personagens)
    p2 = random.choice(personagens)
    v1 = random.randint(0, 20)
    v2 = random.randint(0, 20)
    fase = random.choice(fases)
    vencedor = j1 if v1 >= v2 else j2

    partidas.append({
        "jogador_1": j1,
        "jogador_2": j2,
        "personagem_1": p1,
        "personagem_2": p2,
        "vitorias_1": v1,
        "vitorias_2": v2,
        "fase": fase,
        "vencedor": vencedor
    })


df = pd.DataFrame(partidas)
df.to_csv("src/data/historico.csv", index=False)
