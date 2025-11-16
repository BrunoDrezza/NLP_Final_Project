Segue o README reescrito em tom mais profissional, mantendo a mesma estrutura geral e o conteúdo técnico:

---

# Projeto: FinBERT em Séries Temporais de Bitcoin

Este repositório reúne o projeto final da disciplina de **Processamento de Linguagem Natural (NLP)**, cujo objetivo é investigar se o modelo **FinBERT** – originalmente pré-treinado em **texto financeiro** – é capaz de extrair informação preditiva sobre a **direção diária do preço do Bitcoin (BTC)** quando alimentado exclusivamente com **sequências de retornos numéricos**, codificados como texto.

Em outras palavras, o FinBERT é utilizado aqui como um modelo para séries temporais “disfarçadas” de texto: busca-se avaliar se ele consegue prever se o preço do dia seguinte irá se mover para cima ou não, considerando apenas janelas de retornos recentes.

---

## 1. Objetivo do Projeto

O projeto tem caráter experimental e está mais próximo de um **estudo de comportamento de modelo (“model probe”)** do que de uma proposta de modelo operacional de trading.

### 1.1 Pergunta de pesquisa

> Dado o histórico diário de retornos do Bitcoin, um modelo **FinBERT fine-tunado** em janelas de retornos consegue **prever a direção do preço do dia seguinte** (subida vs. não subida) com desempenho superior a um classificador trivial (por exemplo, classe majoritária)?

### 1.2 Hipótese

A hipótese inicial era:

> A arquitetura pré-treinada do FinBERT poderia capturar algum padrão temporal nos retornos diários e exibir **poder preditivo não trivial** para a direção do dia seguinte.

Os resultados empíricos, na formulação adotada, indicam que:

* o modelo converge para uma estratégia de **sempre prever “alta”**;
* a acurácia obtida é da ordem de **51%**, muito próxima da proporção da classe majoritária;
* não há evidência consistente de previsibilidade out-of-sample.

---

## 2. Metodologia

### 2.1 Dados

* **Fonte:** [`yfinance`](https://pypi.org/project/yfinance/), ticker `BTC-USD`;
* **Período:** de 2014 até a data de execução do script;
* **Frequência:** diária;
* **Variáveis utilizadas:**

  * `btc_price`: preço de fechamento diário (`Close`);
  * `Volume`: volume diário (armazenado, mas não utilizado na versão atual do modelo).

Os dados são armazenados em dois níveis:

* `Data/raw/bitcoin_price_data_raw.csv`: dados brutos (preço e volume);
* `Data/processed/btc_price_data.csv`: série processada utilizada pelo modelo.

### 2.2 Construção dos exemplos

1. A partir do preço de fechamento diário, é calculado o **retorno diário**:

   [
   r_t = \frac{close_t}{close_{t-1}} - 1
   ]

2. Para cada dia ( t ), a partir de um determinado lookback ( L ), é construída uma **janela de retornos**:

   [
   [r_{t-L+1}, \dots, r_t]
   ]

   No experimento padrão, utiliza-se ( L = 30 ) dias.

3. Cada janela é convertida em **sequência textual**, por exemplo:

   ```text
   +0.0050 -0.0123 +0.0033 ...
   ```

4. O **rótulo** associado à janela é definido como:

   * `1` se ( close_{t+1} > close_t ) (preço do dia seguinte maior do que o atual – “up”);
   * `0` caso contrário (“down_or_flat”).

Assim, cada amostra do dataset é composta por:

* **Entrada:** sequência textual de retornos diários em uma janela de ( L ) dias;
* **Saída:** direção do dia seguinte (`up` / `down_or_flat`).

### 2.3 Divisão temporal (sem look-ahead)

Para evitar *data leakage*:

* os dados são ordenados por data;
* as janelas são construídas respeitando a ordem temporal;
* o conjunto é dividido da seguinte forma:

  * **Treino:** primeiros 80% das janelas (período mais antigo);
  * **Teste:** últimos 20% das janelas (período mais recente).

Não há embaralhamento entre treino e teste. O embaralhamento ocorre apenas no **DataLoader de treino**, o que não introduz *look-ahead*.

### 2.4 Modelo

* **Modelo base:** `ProsusAI/finbert` (via `transformers`);
* **Cabeça de classificação:** substituída por uma camada linear de **2 classes** (`up` / `down_or_flat`);
* **Tokenização:** tokenizer do FinBERT, com `max_length = 128`;
* **Tarefa:** classificação binária.

Configuração padrão de treino:

* Otimizador: `AdamW` (`lr = 2e-5`);
* Batch size: 16 (treino) e 64 (teste);
* Épocas: 3;
* Métricas principais: **accuracy** e **F1-weighted**.

---

## 3. Resultados

Em execuções típicas (treino em aproximadamente 80% do histórico e teste nos 20% finais), observam-se resultados semelhantes a:

* **Acurácia em teste:** ~0,51;
* **F1-weighted:** ~0,35;
* **Matriz de confusão:**

  * classe `up`: recall próximo de 1,0;
  * classe `down_or_flat`: recall próximo de 0,0.

Na prática, o modelo converge para uma estratégia de previsão extremamente simples:

> “Prever sempre que o preço irá subir.”

Dado que o dataset é levemente desbalanceado a favor da classe `up` (~51%), a acurácia resulta **marginalmente acima** de um chute aleatório uniforme e equivalente a um classificador de classe majoritária.

### 3.1 Conclusão

* Não foram encontradas **evidências estatisticamente robustas** de poder preditivo out-of-sample para a direção diária do Bitcoin, utilizando o FinBERT alimentado unicamente com janelas de retornos diários;
* O comportamento observado é consistente com a hipótese de que, nesse horizonte diário e com a representação de features utilizada, a série de preços se aproxima de um **passeio aleatório**;
* Do ponto de vista de NLP, o experimento funciona como um **teste de transferência de pré-treino**:

  * o pré-treino em linguagem financeira **não se transfere automaticamente** para uma tarefa puramente numérica representada textualmente;
  * a arquitetura Transformer é capaz de operar sobre sequências numéricas tokenizadas, mas não supera um baseline trivial nesse contexto específico.

---

## 4. Estrutura do Repositório

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

---

## 5. Stack Tecnológico

* **Linguagem:** Python 3.10+
* **Deep Learning / NLP:**

  * `PyTorch`
  * `transformers` (Hugging Face) – modelo `ProsusAI/finbert`
* **Manipulação de dados e métricas:**

  * `pandas`, `numpy`
  * `scikit-learn` (métricas de classificação)
* **Séries temporais:**

  * `yfinance` (coleta de dados de mercado)
* **Visualização:**

  * `matplotlib`, `seaborn`

---

## 6. Como Executar o Projeto

### 6.1 Clonagem do repositório

```bash
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>
```

### 6.2 Criação e ativação do ambiente virtual

```bash
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux/Mac:
source venv/bin/activate
```

### 6.3 Instalação das dependências

```bash
pip install -r requirements.txt
```

---

## 7. Modos de Execução

A estrutura foi pensada para permitir diferentes níveis de uso, desde a execução completa do pipeline até a reutilização apenas de resultados já gerados.

Todas as chamadas são feitas a partir da raiz do projeto, utilizando:

```bash
python -m src.<nome_do_script>
```

### 7.1 Pipeline completo (dados até figuras)

```bash
# 1. Baixar e preparar dados diários de preço do BTC
python -m src.get_bitcoin_data

# 2. Treinar o FinBERT como classificador binário de direção
python -m src.train_finbert_model

# 3. Gerar visualizações (matriz de confusão, curvas de loss/accuracy/F1)
python -m src.plot_finbert_results
```

Ao final, serão gerados:

* modelo salvo em `results/models/finbert_btc_direction_model/`;
* CSVs de métricas em `results/csv/`;
* figuras em `results/figures/`.

### 7.2 Apenas treinamento (dados já processados)

Caso o arquivo `Data/processed/btc_price_data.csv` já exista:

```bash
python -m src.train_finbert_model
```

O modelo e as métricas serão atualizados em `results/`.

### 7.3 Apenas geração de figuras (resultados existentes)

Se os resultados já tiverem sido produzidos em execuções anteriores, garantindo a existência de:

* `results/csv/finbert_btc_direction_history.csv`
* `results/csv/finbert_btc_direction_predictions.csv`
* `results/csv/finbert_btc_direction_classification_report.csv`

é possível gerar ou recriar as figuras com:

```bash
python -m src.plot_finbert_results
```

As imagens serão salvas em `results/figures/`.

---

## 8. Limitações e Possíveis Extensões

Algumas direções naturais para extensão do trabalho:

* Comparação explícita com **modelos clássicos de séries temporais**, como:

  * regressão logística com features `[r_t, r_{t-1}, ...]`;
  * modelos lineares (AR/ARIMA);
  * arquiteturas recorrentes (LSTM) ou Transformers aplicados diretamente a sequências numéricas.

* Alteração do **horizonte de previsão** (por exemplo, semanal ou mensal) e/ou uso de **retornos agregados**.

* Inclusão de **novas variáveis explicativas**:

  * medidas de volatilidade;
  * indicadores técnicos;
  * volume e outros fatores de mercado.

* Integração de **dados textuais**, aproximando o uso do FinBERT de seu domínio original (notícias, relatórios, tweets financeiros), combinando sinais de séries temporais com sinais de sentimento/linguagem.

No estado atual, o repositório é estruturado como um **experimento reproduzível** que ilustra, de forma controlada, que:

> O pré-treino em linguagem natural, isoladamente, **não implica em capacidade preditiva relevante em mercados financeiros** quando o modelo é alimentado apenas com séries numéricas representadas como texto.

---