# GPT-2 Based LLM (MoE + GQA Experiment)

Este repositório contém os códigos-fonte utilizados para um experimento no qual é comparado a arquitetura original do GPT-2 com uma versão moderna inspirada no **Qwen3**, incorporando técnicas como **Grouped-Query Attention (GQA)** e **Mixture of Experts (MoE)**.

Os experimentos foram realizados com recursos limitados (GPU NVIDIA P100 — Kaggle), com foco em eficiência de treinamento e análise prática dos trade-offs entre as duas arquiteturas.

---

## Configuração do Ambiente

Antes de executar os experimentos, é necessário criar um ambiente virtual e instalar as dependências:

```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar a venv
# No Windows:
.venv\Scripts\activate
# No Linux/Mac:
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

---

## Execução dos Experimentos

Os experimentos foram realizados no Kaggle, utilizando uma GPU NVIDIA P100.

Notebook público: [Kaggle — GPT-2 Based LLM Run](https://www.kaggle.com/code/matheushensley/gpt-2-based-llm-run)

Para baixar e preparar os dados:

```bash
python -m scripts.download_data 50
python -m scripts.prepare_data
```

---

## Relatório Técnico

O [relatório técnico](./results/relatorio_tecnico.pdf) completo, contendo:

1. Metodologia detalhada
2. Descrição dos modelos
3. Hiperparâmetros utilizados
4. Métricas (perplexidade, throughput, uso de memória) e análise
5. Discussão dos trade-offs entre GPT-2 e MoE + GQA
6. Trabalhos futuros

## Referências

* Radford, A. et al. (2019). Language Models are Unsupervised Multitask Learners.

* OpenAI — GPT-2 Technical Report

* Qwen Team (2024). Qwen2 and Qwen3 Series.

* Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.

* Sebastian Raschka — LLMs from Scratch.