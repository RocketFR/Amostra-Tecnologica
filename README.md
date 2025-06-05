# Smash Predictor

# 🔮 Amostra Tecnológica: Preditor de Vencedores em Jogos

Um projeto de machine learning que prevê o vencedor em jogos baseado em histórico de partidas, usando Random Forest e Streamlit para a interface.

## 🚀 Funcionalidades
- **Previsão de Vencedores**: 
  - Manual (inserindo dados dos jogadores)
  - Aleatória (seleciona um jogo do histórico)
- **Visualização de Dados**:
  - Gráficos de probabilidade de vitória
  - Desempenho histórico por personagem
- **Editor de Dados**:
  - Adicionar/editar registros no histórico

## 🛠️ Tecnologias
| Componente       | Tecnologia          |
|------------------|---------------------|
| Backend          | Python 3.10+        |
| Machine Learning | Scikit-learn        |
| Interface       | Streamlit           |
| Processamento   | Pandas, NumPy      |
| Visualização    | Matplotlib, Seaborn|

🎮 Como Usar
Previsão Manual:

Preencha os dados dos jogadores na sidebar


Clique em "Prever Vencedor"

Previsão Aleatória:

Clique em "Selecionar Jogadores Aleatórios"

O sistema escolhe um jogo do histórico

Editar Dados:

Na aba "Editor de Dados", adicione ou modifique registros

python -m streamlit run src/app.py
