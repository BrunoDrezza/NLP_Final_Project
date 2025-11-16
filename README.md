
# Projeto: FinBERT em Séries Temporais de Bitcoin

Este repositório contém o projeto final da disciplina de **Processamento de Linguagem Natural (NLP)**.  
O objetivo é testar, de forma controlada, se o modelo **FinBERT** — originalmente treinado para *texto financeiro* — consegue aprender alguma informação preditiva sobre a **direção diária do preço do Bitcoin (BTC)** quando alimentado apenas com **sequências de retornos numéricos**, codificados como texto.

> Em outras palavras: aqui o FinBERT é “abusado” como modelo de séries temporais, e não como modelo de linguagem. A pergunta é: **ele consegue prever se o preço de amanhã sobe ou cai, só olhando para os últimos retornos?**

---

## 🎯 Objetivo do Projeto

Este projeto está mais na linha **“experimento científico / probe de modelo”** do que “modelo de trading real”.

### Pergunta de pesquisa

> Dado o histórico diário de retornos do Bitcoin, um modelo **FinBERT fine-tunado** em janelas de retornos consegue **prever a direção do preço do dia seguinte** (subir vs. não subir) melhor do que um classificador trivial?

### Hipótese (spoiler: ela cai por terra)

A hipótese inicial era:

> “Talvez a arquitetura pré-treinada do FinBERT consiga capturar algum padrão temporal nos retornos diários e exiba **poder preditivo não trivial** para a direção do próximo dia.”

O experimento mostra que, **na forma como o problema foi formulado**, o modelo:

- aprende a **sempre prever “alta”**,  
- alcança cerca de **51% de acurácia**, muito próximo da **classe majoritária**,  
- e **não** apresenta evidências robustas de previsibilidade out-of-sample.

---

## 🧠 Metodologia

### 1. Dados

- **Fonte:** [`yfinance`](https://pypi.org/project/yfinance/) – ticker `BTC-USD`
- **Período:** de 2014 até a data de execução do script
- **Granularidade:** diária
- **Variáveis utilizadas:**
  - `btc_price` – preço de fechamento (`Close`)
  - `Volume` – volume diário (apenas armazenado; o modelo atual usa apenas preço)

Os dados são salvos em dois níveis:

- `Data/raw/bitcoin_price_data_raw.csv` – dados crus de preço/volume  
- `Data/processed/btc_price_data.csv` – versão processada usada pelo modelo

### 2. Construção dos exemplos

1. A partir do preço de fechamento diário, é calculado o **retorno diário**:

   \[
   r_t = \frac{close_t}{close_{t-1}} - 1
   \]

2. Para cada dia \( t \) (a partir de um certo `lookback`), é criada uma **janela de retornos**:

   \[
   [r_{t-L+1}, \dots, r_t]
   \]

   com `L = 30` dias (por padrão).

3. A janela é transformada em **texto**, por exemplo:

   ```text
   +0.0050 -0.0123 +0.0033 ...
````

4. O **rótulo** associado a essa janela é:

   * `1` se `close_{t+1} > close_t` (dia seguinte sobe),
   * `0` caso contrário (`down_or_flat`).

Assim, cada amostra é:

* **Entrada:** sequência textual de 30 retornos diários
* **Saída:** direção do dia seguinte (`up` / `down_or_flat`)

### 3. Split temporal (sem look-ahead)

Para evitar *data leakage*:

* Os dados são ordenados por data.
* As janelas são construídas respeitando a ordem temporal.
* O dataset final é dividido como:

  * **Treino:** primeiros **80%** das janelas (período mais antigo)
  * **Teste:** últimos **20%** das janelas (período mais recente)

Não há embaralhamento entre treino e teste; apenas o **DataLoader de treino** embaralha **dentro** do conjunto de treino, o que não introduz *look-ahead*.

### 4. Modelo

* **Modelo base:** `ProsusAI/finbert` (via `transformers`)
* **Cabeça de classificação:** substituída por uma camada linear de **2 classes** (`up` / `down_or_flat`)
* **Tokenização:** FinBERT tokenizer, com `max_length = 128`
* **Tarefa:** classificação binária

Treinamos o modelo com:

* Otimizador: `AdamW` (`lr = 2e-5`)
* Batch size: 16 (treino), 64 (teste)
* Épocas: 3
* Métricas: **accuracy** e **F1-weighted**

---

## 📈 Resultados Principais

Em uma execução típica (treino até ~80% do histórico, teste nos ~20% mais recentes), observa-se:

* **Acurácia de teste:** ~0.51
* **F1-weighted:** ~0.35
* **Matriz de confusão:**

  * Classe `up` – recall ≈ 1.0
  * Classe `down_or_flat` – recall ≈ 0.0

Ou seja, o modelo converge para uma política extremamente simples:

> **“Sempre diga que o preço vai subir.”**

Como o dataset é levemente desbalanceado para `up` (~51%), a acurácia fica **apenas marginalmente** acima de um chute uniformemente aleatório, e efetivamente igual a um classificador trivial de classe majoritária.

### Conclusão (estilo paper)

* **Não encontramos evidências estatisticamente convincentes** de poder preditivo out-of-sample para a direção do dia seguinte do Bitcoin usando FinBERT alimentado apenas com janelas de retornos diários.
* O comportamento do modelo sugere que, nesse horizonte diário e com essa representação de features, a série de preços se comporta como um **passeio aleatório** para o modelo — consistente com hipóteses de eficiência de mercado em escalas curtas.
* Do ponto de vista de NLP, o experimento é um “stress test” interessante:

  * O pré-treino em linguagem financeira **não se transfere** automaticamente para uma tarefa puramente numérica disfarçada de texto.
  * A arquitetura Transformer ainda funciona como um modelo de série temporal, mas não supera um baseline de classe majoritária neste setup.

---

## 🧱 Estrutura do Repositório

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
├── Notebooks/   # (opcional, uso exploratório)
├── requirements.txt
└── README.md
```

---

## 🔧 Stack Tecnológico

* **Linguagem:** Python 3.10+
* **Deep Learning / NLP:**

  * `PyTorch`
  * `transformers` (Hugging Face) – FinBERT (`ProsusAI/finbert`)
* **Dados & Métricas:**

  * `pandas`, `numpy`
  * `scikit-learn` (métricas de classificação)
* **Séries Temporais (dados):**

  * `yfinance`
* **Visualização:**

  * `matplotlib`, `seaborn`

---

## 🚀 Como Executar o Projeto

### 0. Clonar o repositório

```bash
git clone https://github.com/<seu-usuario>/<seu-repositorio>.git
cd <seu-repositorio>
```

### 1. Criar e ativar ambiente virtual

```bash
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

---

## 🧪 Modos de Execução

A ideia é deixar tudo modular. Você pode:

1. **Baixar dados + treinar modelo + gerar figuras** (pipeline completo)
2. **Treinar apenas o modelo** (caso já tenha os dados)
3. **Gerar apenas as figuras** (caso já tenha os CSVs de resultados, por exemplo, vindos do repositório)

Todos os scripts são chamados via `python -m src.<nome_do_script>` a partir da raiz do projeto.

### 🔁 1) Pipeline completo (do zero até as figuras)

```bash
# 1. baixar e preparar os dados diários de preço do BTC
python -m src.get_bitcoin_data

# 2. treinar o FinBERT como classificador binário de direção
python -m src.train_finbert_model

# 3. gerar métricas visuais (matriz de confusão, curvas de loss/acc/F1)
python -m src.plot_finbert_results
```

Ao final, você terá:

* modelo salvo em `results/models/finbert_btc_direction_model/`
* CSVs de métricas em `results/csv/`
* figuras em `results/figures/`

### 🧠 2) Apenas treinar (dados já baixados)

Se você já tiver o arquivo `Data/processed/btc_price_data.csv` (porque fez o passo 1 ou fez o download manual), basta:

```bash
python -m src.train_finbert_model
```

Os resultados (modelo + CSVs) serão sobrescritos/atualizados na pasta `results/`.

### 📊 3) Apenas gerar as figuras (usar resultados existentes)

Se você não quiser gastar tempo treinando de novo e só quer as imagens:

* Certifique-se de que existem:

  * `results/csv/finbert_btc_direction_history.csv`
  * `results/csv/finbert_btc_direction_predictions.csv`
  * `results/csv/finbert_btc_direction_classification_report.csv`

Então rode:

```bash
python -m src.plot_finbert_results
```

As figuras serão recriadas (ou sobrescritas) em `results/figures/`.

---

## 📌 Limitações e Extensões Futuras

Algumas ideias claras de extensões, caso alguém queira continuar o projeto:

* Comparar explicitamente com **baselines de séries temporais**:

  * regressão logística com features `[r_t, r_{t-1}, ...]`,
  * modelos lineares AR,
  * LSTM/Transformer “clássicos” em cima de retornos.
* Mudar o **horizonte de previsão** (semanal, mensal) e/ou usar **retornos agregados**.
* Testar versões do modelo que usem **mais features** (volatilidade, volume, indicadores técnicos).
* Integrar, de fato, **NLP de notícias/tweets** para fazer algo mais próximo do uso original do FinBERT.

No estado atual, o repositório foca em ser um **experimento limpo e reproduzível** para mostrar que:

> pré-treino em linguagem natural, por si só, **não cria um oráculo de mercado** quando alimentado apenas com números.

---

```
::contentReference[oaicite:0]{index=0}
```