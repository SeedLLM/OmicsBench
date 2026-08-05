```text
.
├── data
│   ├── FunctionEC_212.json
│   ├── Modification_101.json
│   ├── NoncodingRNAFamily_215.json
│   ├── cpd_111.json
│   ├── emp-H3_197.json
│   ├── pd_108.json
│   ├── tf-h_103.json
│   └── tf-m_113.json
├── evaluation
│   ├── ec_labels.json
│   ├── evaluate_llm_judge.py
│   ├── evaluate_llm_judge.sh
│   ├── evaluate_metric.py
│   ├── evaluate_metric.sh
│   └── register_tasks.json
├── README.md
└── generate.py
```

**OmicsBench** is a pioneering benchmark designed to evaluate the scientific reasoning capabilities of Large Language Models (LLMs) in the context of multi-omics sequence analysis. Unlike traditional benchmarks that focus on black-box classification and regression metrics, OmicsBench requires models to provide traceable evidence chains, bridging the gap between prediction and genuine biological understanding.

<img width="1994" height="355" alt="image" src="https://github.com/user-attachments/assets/f708fddd-c5aa-4a9a-8b68-cadef8dbfa8c" />



## 🚀 Overview

Multi-omics sequences (DNA, RNA, proteins) encode complex biological mechanisms essential for understanding disease, designing therapeutics, and automated scientific discovery. While LLMs have shown promise in these areas, existing evaluations often fail to distinguish between shortcut learning (statistical pattern matching) and true scientific reasoning.

OmicsBench addresses this by:
-   Comprising **1,160 expert-validated questions**.
-   Covering **six biologically coherent tasks**.
-   Spanning the central dogma: **DNA regulation**, **RNA processing**, and **Protein function**.
-   Evaluating traceable reasoning chains using instance-specific rubrics.

<img width="1546" height="845" alt="image" src="https://github.com/user-attachments/assets/a813b8d6-5dfc-4d36-b3d9-417dd13528e4" />


## 🧬 Tasks

OmicsBench is organized along the sequential logic of multi-omics information processing:

| Category | Task | Type | Metric | N | % |
|:---------|:-----|:-----|:-------|----:|------:|
| **DNA Regulation** | Epigenetic Mark Prediction | Binary | MCC | 197 | 17.0% |
| | Promoter Detection | Binary | MCC | 219 | 18.9% |
| | Transcription Factor Binding Site Prediction | Binary | MCC | 216 | 18.6% |
| **RNA Processing** | RNA Modification Prediction | Multi-label | AUC | 101 | 8.7% |
| | Non-coding RNA Classification | Multi-class | Acc | 215 | 18.5% |
| **Protein Function** | Enzyme Function Prediction | Multi-label | F-max | 212 | 18.3% |
| **Total** | | | | **1,160** | **100.0%** |

### 1. DNA Regulation
-   Identifying epigenetic marks
-   Promoter region analysis
-   Transcription factor binding sites

### 2. RNA Processing
-   Characterizing RNA modifications
-   Non-coding RNA analysis

### 3. Protein Function
-   Annotating enzyme functions


## 🛠️ Methodology

To ensure high-quality and scalable reasoning traces, OmicsBench utilizes a **multi-agent synthesis framework**. Tool-augmented bio-agents query biological databases, perform sequence alignments, and retrieve literature evidence to automatically curate reasoning chains.

All questions and solutions undergo a rigorous **two-tier validation process**:
1.  Machine-based checks.
2.  Expert reviews.

## 🔍 Results

| Model | DNA (Predictive) | | | RNA (Predictive) | | Prot. | Avg | DNA (Reasoning) | | | RNA (Reasoning) | | Prot. | Avg |
|-------|------------------|-|-|---------------|---|-------|-----|----------------|-|-|---------------|---|-------|-------|
| | EMP<br>(MCC) | Prom<br>(MCC) | TFBS<br>(MCC) | Mod<br>(AUC) | ncRNA<br>(Acc) | EC<br>(Fmax) | Rank<br>↓ | EMP<br>(%) | Prom<br>(%) | TFBS<br>(%) | Mod<br>(%) | ncRNA<br>(%) | EC<br>(%) | Recall<br>↑ |
| **Proprietary LLMs** |
| Claude-Sonnet-4.5 | 3.56 | 28.58 | 12.50 | 53.42 | 12.56 | 8.04 | 7.67 | 23.27 | 25.60 | 12.88 | 20.63 | 9.49 | 4.77 | 16.11 |
| Gemini-3-Pro | -10.22 | 16.61 | 2.32 | 51.57 | 24.19 | 24.68 | 9.67 | 4.74 | 22.21 | 24.96 | 8.17 | 24.31 | 12.60 | 16.17 |
| GLM-4.7 | -5.29 | 12.16 | 6.96 | 50.90 | 7.44 | 8.94 | 12.50 | 4.44 | 23.13 | 12.43 | 13.53 | 7.81 | 3.35 | 10.78 |
| GPT-5.2 | 14.38 | 15.97 | -0.81 | 53.17 | 3.72 | 8.62 | 11.67 | 4.70 | 20.00 | 21.77 | 7.26 | 8.74 | 4.07 | 11.09 |
| Grok-4 | -3.50 | 30.67 | 2.09 | 50.28 | 13.95 | 23.39 | 9.33 | 10.79 | 30.16 | 27.46 | 10.81 | 34.04 | 13.34 | 21.10 |
| Kimi-K2 | 9.65 | 0.94 | 4.85 | 47.24 | 4.65 | 16.58 | 12.00 | 5.29 | 16.23 | 9.23 | 12.95 | 2.40 | 9.03 | 9.19 |
| Qwen3-Max | 8.36 | 21.35 | -0.98 | 53.10 | 19.07 | 10.91 | 9.00 | 6.30 | 28.89 | 25.39 | 29.95 | 16.07 | 4.55 | 18.53 |
| **Open-Source LLMs** |
| DeepSeek-V3.2 | 1.28 | 23.78 | 8.91 | 52.25 | 9.77 | 11.68 | 8.67 | 12.27 | 25.79 | 20.82 | 5.86 | 6.65 | 4.31 | 12.62 |
| GPT-OSS-120B | 20.96 | 18.98 | 0.17 | 52.48 | 8.84 | 11.46 | 9.17 | 4.74 | 22.37 | 24.17 | 19.97 | 18.52 | 3.98 | 15.63 |
| Llama-4-Maverick | 14.72 | 17.53 | 4.89 | 53.50 | 6.98 | 10.42 | 8.83 | 2.16 | 11.36 | 3.81 | 2.23 | 2.20 | 2.67 | 4.07 |
| Qwen3-235B | 1.05 | 2.97 | 0.68 | 54.12 | 5.12 | 7.82 | 12.17 | 9.56 | 28.19 | 22.08 | 25.33 | 7.99 | 3.40 | 16.09 |
| **Scientific LLMs** |
| ChatMultiOmics | 10.44 | 20.66 | 21.83 | 59.72 | 83.26 | 23.12 | 4.33 | 1.27 | 4.07 | 1.59 | 0.00 | 20.80 | 2.65 | 5.06 |
| ChatNT | 86.86 | 40.76 | 12.46 | - | - | - | 3.00 | 0.00 | 0.00 | 0.25 | - | - | - | 0.08 |
| Intern-S1 | -0.11 | 19.24 | 21.81 | 51.36 | 11.63 | 13.56 | 9.00 | 2.33 | 20.45 | 12.39 | 12.05 | 2.44 | 3.67 | 8.89 |
| Intern-S1-Pro | 20.61 | 47.93 | 43.95 | 52.27 | 17.67 | 39.53 | 3.50 | 0.00 | 0.23 | 0.00 | 0.00 | 3.99 | 10.23 | 2.41 |
| NatureLM | -6.66 | 32.41 | 10.87 | 52.04 | 0.00 | 14.76 | 9.83 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.35 | 0.23 |
| SciReasoner | 29.67 | 31.29 | 44.37 | 50.52 | 6.51 | 77.41 | 5.67 | 3.17 | 2.00 | 0.00 | 0.99 | 1.05 | 1.11 | 1.39 |
