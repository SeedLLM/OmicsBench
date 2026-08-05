#!/bin/bash
# Configuration
INPUT_DIR="model_predictions"
OUTPUT_DIR="evaluation_results/omics_metric"
SENTIMENT_MODEL_PATH="models/witter-roberta-base-sentiment-latest"

# Iterate over JSON files in model_predictions.
for json_file in "$INPUT_DIR"/*.json; do
  # Check that the file exists.
  if [ ! -f "$json_file" ]; then
    echo "⚠️  JSON file not found: $INPUT_DIR"
    continue
  fi
  
  # Use the filename without its extension as model_name.
  model_name=$(basename "$json_file" .json)
  
  echo "========================================"
  echo "Processing model: $model_name"
  echo "========================================"
  
  # Run evaluation.
  python evaluate_metric.py \
    --model_name "$model_name" \
    --OMICS All \
    --input_file_path "$json_file" \
    --output_dir "$OUTPUT_DIR" \
    --sentiment_model_path "$SENTIMENT_MODEL_PATH"
  
  echo ""
done

echo "✓ Evaluation completed for all models."
