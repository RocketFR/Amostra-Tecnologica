# 🔮 SmashPredictor
**Sistema de Predição de Resultados para Torneios de Super Smash Bros**

---

## 📌 Descrição
O **SmashPredictor** é um sistema que utiliza algoritmos de Machine Learning para prever, com base em dados históricos, quem tem mais chances de vencer uma partida de *Super Smash Bros*. A aplicação possui uma interface interativa via **Streamlit**, onde o usuário pode inserir os dados de dois jogadores e receber o "Palpite da IA", com visualização gráfica das probabilidades.

---

## 🎯 Objetivo
Facilitar a organização de torneios com análises inteligentes, fornecendo previsões baseadas em dados históricos de confrontos anteriores.

---

## 👥 Público-alvo
- Organizadores de torneios de Smash Bros
- Jogadores competitivos e casuais
- Entusiastas de IA aplicada a jogos
- Alunos em projetos de IA, Jogos e Ciência de Dados

---

## 🧠 Algoritmos Utilizados
- `RandomForestClassifier` (da biblioteca `scikit-learn`)
- `LabelEncoder` para codificação de personagens e fases
- Lógica de classificação binária (`vencedor_bin`) para treinar o modelo com base em vitórias do jogador 1

---

## 🛠️ Tecnologias Usadas
- **Python 3.10+**
- **pandas**
- **scikit-learn**
- **streamlit**
- **matplotlib**
- **seaborn**

---

## 📁 Estrutura do Projeto

/
├── src/
│ ├── data/
│ │ └── historico.csv <- Arquivo com dados históricos gerados
│ ├── models/
│ │ └── predictor.py <- Funções de treino e predição
│ ├── utils/
│ │ └── preprocessing.py <- Carregamento e transformação dos dados
│ ├── app.py <- Interface interativa com Streamlit
│ └── main.py <- Execução via terminal (modo CLI)
├── Script gerar dados.py <- Script para gerar partidas fictícias
├── requirements.txt <- Dependências do projeto
└── README.md <- Este arquivo
