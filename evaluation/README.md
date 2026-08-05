# OmicsBench Evaluation Scripts

This directory contains evaluation scripts for the OmicsBench benchmark.

## Overview

- **evaluate_metric.py**: Automated metrics evaluation
- **evaluate_llm_judge.py**: LLM-as-a-judge evaluation
- **evaluate_metric.sh**: Batch processing for metrics
- **evaluate_llm_judge.sh**: Batch processing for LLM judge

## Quick Start

### Metrics Evaluation
```bash
cd evaluation
bash evaluate_metric.sh
```

### LLM Judge Evaluation
```bash
cd evaluation
# Edit evaluate_llm_judge.sh to set your API_KEY
bash evaluate_llm_judge.sh
```
