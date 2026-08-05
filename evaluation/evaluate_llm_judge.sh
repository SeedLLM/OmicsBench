#!/bin/bash

# Configuration
INPUT_DIR="model_predictions"           # Directory containing model predictions
RUBRIC_DIR="rubrics"                    # Directory containing rubric files
OUTPUT_DIR="evaluation_results/llm_judge"
JUDGE_MODEL="qwen-max"
API_KEY="your-api-key-here"
API_URL="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

# Run evaluation
python evaluate_llm_judge.py \
  --input_dir "$INPUT_DIR" \
  --rubric_dir "$RUBRIC_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --judge_model "$JUDGE_MODEL" \
  --api_key "$API_KEY" \
  --api_url "$API_URL" \
  --max_workers 10 \
  --max_retries 5
