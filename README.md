# 📈 Projeto: FinBERT em Séries Temporais de Bitcoin

> **Análise experimental de previsibilidade da direção diária do Bitcoin usando FinBERT aplicado a séries numéricas codificadas como texto.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/HF-Transformers-yellow)
![Status](https://img.shields.io/badge/status-Experimental-orange)

---

## 📚 Sumário

- [Visão Geral](#-visão-geral)
- [Objetivo do Projeto](#-objetivo-do-projeto)
  - [Pergunta de Pesquisa](#-pergunta-de-pesquisa)
  - [Hipótese](#-hipótese)
- [Metodologia](#-metodologia)
  - [Dados](#-dados)
  - [Construção dos Exemplos](#-construção-dos-exemplos)
  - [Divisão Temporal](#-divisão-temporal-sem-look-ahead)
  - [Modelo](#-modelo)
- [Resultados](#-resultados)
  - [Conclusão](#-conclusão)
- [Estrutura do Repositório](#-estrutura-do-repositório)
  - [Gerenciamento de Caminhos (`src/paths.py`)](#-gerenciamento-de-caminhos-srcpathspy)
- [Stack Tecnológico](#-stack-tecnológico)
- [Como Executar o Projeto](#-como-executar-o-projeto)
- [Modos de Execução](#-modos-de-execução)
- [Limitações e Possíveis Extensões](#-limitações-e-possíveis-extensões)

---

## 🔎 Visão Geral

Este repositório contém o projeto final da disciplina de **Processamento de Linguagem Natural (NLP)**.

A ideia central é investigar se o modelo **FinBERT** – originalmente pré-treinado em **texto financeiro** – consegue extrair **informação preditiva** sobre a **direção diária do preço do Bitcoin (BTC)** quando alimentado exclusivamente com **sequências de retornos numéricos**, codificados como texto.

Em termos práticos, tratamos o FinBERT como um modelo aplicado a **séries temporais “disfarçadas” de texto**: o objetivo é verificar se ele consegue prever se o preço do dia seguinte irá subir ou não, considerando apenas uma janela recente de retornos.

---

## 🎯 Objetivo do Projeto

O projeto tem caráter **experimental** e se aproxima mais de um **estudo de comportamento de modelo (“model probe”)** do que de uma proposta de modelo operacional de trading.

### ❓ Pergunta de Pesquisa

> Dado o histórico diário de retornos do Bitcoin, um modelo **FinBERT fine-tunado** em janelas de retornos consegue **prever a direção do preço do dia seguinte** (subida vs. não subida) com desempenho superior a um classificador trivial (por exemplo, classe majoritária)?

### 💡 Hipótese

A hipótese inicial era que:

> A arquitetura pré-treinada do FinBERT poderia capturar algum padrão temporal nos retornos diários e exibir **poder preditivo não trivial** para a direção do dia seguinte.

Os resultados empíricos indicam que, na formulação deste experimento:

- o modelo converge para uma estratégia de **sempre prever “alta”**;
- a acurácia obtida é da ordem de **51%**, muito próxima da proporção da classe majoritária;
- não há evidência consistente de **previsibilidade out-of-sample**.

---

## 🧪 Metodologia

### 📊 Dados

- **Fonte:** [`yfinance`](https://pypi.org/project/yfinance/), ticker `BTC-USD`
- **Período:** de 2014 até a data de execução do script
- **Frequência:** diária
- **Variáveis:**
  - `btc_price`: preço de fechamento diário (`Close`)
  - `Volume`: volume diário (armazenado, mas não utilizado na versão atual do modelo)

Arquivos gerados:

- `Data/raw/bitcoin_price_data_raw.csv` – dados brutos (preço e volume)
- `Data/processed/btc_price_data.csv` – série processada utilizada pelo modelo

### 🧱 Construção dos Exemplos

1. A partir do preço de fechamento diário, é calculado o **retorno diário**:

   \[
   r_t = \frac{close_t}{close_{t-1}} - 1
   \]

2. Para cada dia \( t \), a partir de um determinado **lookback** \( L \), é construída uma **janela de retornos**:

   \[
   [r_{t-L+1}, \dots, r_t]
   \]

   No experimento padrão, utiliza-se **\( L = 30 \)** dias.

3. Cada janela é convertida em **sequência textual**, por exemplo:

   ```text
   +0.0050 -0.0123 +0.0033 ...
   ```

4. O **rótulo** associado à janela é definido como:

   - `1` se \( close_{t+1} > close_t \) (preço do dia seguinte maior – “up”);
   - `0` caso contrário (“down_or_flat”).

Assim, cada amostra do dataset é composta por:

- **Entrada:** sequência textual de retornos diários em uma janela de \( L \) dias  
- **Saída:** direção do dia seguinte (`up` / `down_or_flat`)

### 🕒 Divisão Temporal (sem *look-ahead*)

Para evitar *data leakage*:

- os dados são ordenados por data;
- as janelas são construídas respeitando a ordem temporal;
- o conjunto é dividido em:
  - **Treino:** primeiros 80% das janelas (período mais antigo)
  - **Teste:** últimos 20% das janelas (período mais recente)

Não há embaralhamento entre treino e teste. O *shuffle* ocorre apenas no **DataLoader de treino**, o que não introduz *look-ahead*.

### 🤖 Modelo

- **Modelo base:** `ProsusAI/finbert` (via `transformers`)
- **Cabeça de classificação:** camada linear de **2 classes** (`up` / `down_or_flat`)
- **Tokenização:** tokenizer do FinBERT (`max_length = 128`)
- **Tarefa:** classificação binária

Configuração padrão de treino:

- Otimizador: `AdamW` (`lr = 2e-5`)
- Batch size: 16 (treino) e 64 (teste)
- Épocas: 3
- Métricas principais: **accuracy** e **F1-weighted**

---

## 📊 Resultados

Em execuções típicas (treino em ~80% do histórico e teste nos 20% finais):

- **Acurácia em teste:** ~0,51  
- **F1-weighted:** ~0,35  
- **Matriz de confusão:**
  - classe `up`: *recall* próximo de 1,0
  - classe `down_or_flat`: *recall* próximo de 0,0

Na prática, o modelo converge para uma regra extremamente simples:

> **“Prever sempre que o preço irá subir.”**

Como o dataset é levemente desbalanceado a favor da classe `up` (~51%), a acurácia fica **marginalmente acima** de um chute aleatório e equivalente a um classificador de classe majoritária.

### ✅ Conclusão

- Não foram encontradas **evidências estatisticamente robustas** de poder preditivo out-of-sample para a direção diária do Bitcoin usando FinBERT alimentado apenas com janelas de retornos diários;
- O comportamento observado é consistente com a hipótese de que, nesse horizonte diário e com essa representação de features, a série de preços se aproxima de um **passeio aleatório**;
- Do ponto de vista de NLP, o experimento funciona como um **teste de transferência de pré-treino**:
  - o pré-treino em linguagem financeira **não se transfere automaticamente** para uma tarefa puramente numérica representada como texto;
  - a arquitetura Transformer consegue operar sobre sequências numéricas tokenizadas, mas **não supera um baseline trivial** neste contexto específico.

---

## 📁 Estrutura do Repositório

```text
NLP_FINAL_PROJECT/
├── Data/
│   ├── raw/
│   │   └── bitcoin_price_data_raw.csv
│   └── processed/
│       └── btc_price_data.csv
├── results/
│   ├── csv/
│   │   ├── finbert_btc_direction_history.csv
│   │   ├── finbert_btc_direction_predictions.csv
│   │   └── finbert_btc_direction_classification_report.csv
│   ├── figures/
│   │   ├── confusion_matrix_btc_direction.png
│   │   ├── train_loss_per_epoch.png
│   │   ├── test_accuracy_per_epoch.png
│   │   └── test_f1_per_epoch.png
│   └── models/
│       └── finbert_btc_direction_model/
├── src/
│   ├── __init__.py
│   ├── paths.py
│   ├── get_bitcoin_data.py
│   ├── train_finbert_model.py
│   └── plot_finbert_results.py
├── Notebooks/          # uso exploratório (opcional)
├── requirements.txt
└── README.md
```

### 📂 Gerenciamento de Caminhos (`src/paths.py`)

O arquivo `src/paths.py` centraliza todos os **paths importantes do projeto** e garante que as pastas necessárias existam antes da execução dos scripts.

Trecho principal:

```python
from pathlib import Path

# Pasta raiz do projeto (onde está requirements.txt, Data/, results/, etc.)
BASE_DIR = Path(__file__).resolve().parents[1]

# Data
DATA_DIR = BASE_DIR / "Data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Results
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
CSV_DIR = RESULTS_DIR / "csv"
FIGURES_DIR = RESULTS_DIR / "figures"

# Garante que as pastas existem
for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, CSV_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
```

Isso permite, por exemplo, que outros módulos utilizem:

```python
from src.paths import RAW_DIR, PROCESSED_DIR, MODELS_DIR, CSV_DIR, FIGURES_DIR
```

sem precisar hardcodar caminhos, mantendo o projeto **portável e organizado**.

---

## 🧰 Stack Tecnológico

- **Linguagem:** Python 3.10+
- **Deep Learning / NLP:**
  - `PyTorch`
  - `transformers` (Hugging Face) – modelo `ProsusAI/finbert`
- **Manipulação de dados e métricas:**
  - `pandas`, `numpy`
  - `scikit-learn` (métricas de classificação)
- **Séries temporais:**
  - `yfinance` (coleta de dados de mercado)
- **Visualização:**
  - `matplotlib`, `seaborn`

---

## 🚀 Como Executar o Projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>
```

### 2. Criar e ativar o ambiente virtual

```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

---

## 🧪 Modos de Execução

Todas as chamadas abaixo assumem que você está na **raiz** do projeto e usa:

```bash
python -m src.<nome_do_script>
```

### 🔁 1. Pipeline completo (dados → modelo → figuras)

```bash
# 1. Baixar e preparar dados diários de preço do BTC
python -m src.get_bitcoin_data

# 2. Treinar o FinBERT como classificador binário de direção
python -m src.train_finbert_model

# 3. Gerar visualizações (matriz de confusão, curvas de loss/accuracy/F1)
python -m src.plot_finbert_results
```

Ao final, serão gerados:

- modelo salvo em `results/models/finbert_btc_direction_model/`;
- CSVs de métricas em `results/csv/`;
- figuras em `results/figures/`.

### 🧱 2. Apenas treinamento (dados já processados)

Se `Data/processed/btc_price_data.csv` já existir:

```bash
python -m src.train_finbert_model
```

O modelo e as métricas serão atualizados em `results/`.

### 📈 3. Apenas geração de figuras (resultados existentes)

Garantindo a existência de:

- `results/csv/finbert_btc_direction_history.csv`
- `results/csv/finbert_btc_direction_predictions.csv`
- `results/csv/finbert_btc_direction_classification_report.csv`

é possível gerar/recriar as figuras com:

```bash
python -m src.plot_finbert_results
```

As imagens serão salvas em `results/figures/`.

---

## ⚠️ Limitações e Possíveis Extensões

Algumas direções naturais para extensão do trabalho:

- Comparar explicitamente com **modelos clássicos de séries temporais**, como:
  - regressão logística com features `[r_t, r_{t-1}, ...]`;
  - modelos lineares (AR/ARIMA);
  - arquiteturas recorrentes (LSTM) ou Transformers aplicados diretamente a sequências numéricas.
- Alterar o **horizonte de previsão** (por exemplo, semanal ou mensal) e/ou usar **retornos agregados**.
- Incluir **novas variáveis explicativas**:
  - medidas de volatilidade;
  - indicadores técnicos;
  - volume e outros fatores de mercado.
- Integrar **dados textuais**, aproximando o uso do FinBERT de seu domínio original (notícias, relatórios, tweets financeiros), combinando sinais de séries temporais com sinais de sentimento/linguagem.

No estado atual, o repositório é estruturado como um **experimento reproduzível** que ilustra, de forma controlada, que:

> O pré-treino em linguagem natural, isoladamente, **não implica em capacidade preditiva relevante em mercados financeiros** quando o modelo é alimentado apenas com séries numéricas representadas como texto.
