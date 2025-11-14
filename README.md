# Projeto: Análise de Sentimento com FinBERT para Previsão de Preços de Bitcoin

Este repositório contém o projeto final para a disciplina de Processamento de Linguagem Natural (NLP). O objetivo é investigar se o sentimento extraído de notícias financeiras, usando o modelo **FinBERT**, pode melhorar a acurácia de um modelo de previsão de séries temporais (**LSTM**) para o preço do Bitcoin (BTC).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.30%2B-yellow.svg)

---

## 🎯 Objetivo do Projeto

Este projeto se enquadra na trilha **"Deep Learning Researcher"**.

O objetivo central não é criar um "bot trader" lucrativo, mas sim **investigar e medir cientificamente** se a incorporação de *features* de sentimento de NLP pode adicionar valor preditivo a um modelo de previsão de séries temporais.

### 💡 Hipótese

> A hipótese central é que um modelo híbrido (LSTM + Sentimento) treinado com dados de preço e dados de sentimento de notícias terá um **erro de previsão (RMSE) menor** do que um modelo LSTM ingênuo (Baseline) treinado apenas com dados históricos de preço.

---

## 🛠️ Arquitetura e Metodologia

O projeto é um sistema híbrido de duas etapas:

1.  **Engenharia de Features de NLP:**
    * As notícias sobre Bitcoin são coletadas e processadas.
    * O modelo **FinBERT** (um BERT pré-treinado em dados financeiros) é usado para analisar o sentimento de cada título de notícia, gerando um score (`positivo`, `negativo`, `neutro`).
    * Esses scores são agregados diariamente (ex: pela média) para criar uma única *feature* de sentimento por dia.

2.  **Previsão de Série Temporal:**
    * Dois modelos LSTM são treinados e comparados:
    * **Modelo A (Baseline):** Um LSTM que usa *apenas* o histórico de preços (ex: `[preço_d-2, preço_d-1, preço_d]`) para prever o `preço_d+1`.
    * **Modelo B (Híbrido):** O *mesmo* LSTM que usa o histórico de preços **E** o histórico de sentimento (ex: `[preço_d-2, sentimento_d-2]`, `[preço_d-1, sentimento_d-1]`, etc.) para prever o `preço_d+1`.

O fluxo de dados segue o seguinte pipeline:



---

## 📊 Datasets Utilizados

1.  **Dados de Preço (BTC):**
    * **Fonte:** `yfinance` (Yahoo Finance)
    * **Ticker:** `BTC-USD`
    * **Granularidade:** Diária (Open, High, Low, Close, Volume)

2.  **Dados de Texto (Notícias):**
    * **Fonte:** Kaggle
    * **Dataset:** [Bitcoin - News articles text corpora](https://www.kaggle.com/datasets/thedevastator/bitcoin-news-articles-text-corpora)
    * **Descrição:** Um corpus de artigos de notícias sobre Bitcoin, contendo colunas como `published date`, `title` e `language`.
    * **Pré-processamento:** Foram utilizados apenas artigos com `language == 'en'` e a análise foi focada no `title` para extração de sentimento.

---

## 🔧 Tecnologia e Ferramentas

Este projeto utiliza o ecossistema PyData e ferramentas de Deep Learning:

* **Core:** `Python 3.10`
* **Deep Learning:** `PyTorch` (para definição do modelo LSTM)
* **NLP:** `Transformers (Hugging Face)` (para carregar e usar o FinBERT)
* **Manipulação de Dados:** `Pandas` e `NumPy`
* **Coleta de Dados:** `yfinance`
* **Métricas e Normalização:** `scikit-learn` (para `MinMaxScaler` e `RMSE`)
* **Ambiente:** `Jupyter Notebook` / `VS Code`

---

## 🚀 Como Executar o Projeto

Para replicar os resultados, siga os passos abaixo:

**1. Clone o repositório:**
```bash
git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
cd seu-repositorio