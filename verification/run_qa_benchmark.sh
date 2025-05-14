#!/bin/bash
# Bash script to run QA benchmark

# Change to the benchmark directory
cd benchmark/

# Set API key from config.json or from environment
CONFIG_PATH="../data/config.json"
if [ -f "$CONFIG_PATH" ]; then
    API_KEY=$(grep -o '"api_key": *"[^"]*"' "$CONFIG_PATH" | grep -o '"[^"]*"$' | tr -d '"')
    export OPENAI_API_KEY=$API_KEY
fi

# Run the benchmark script
python ./qa_benchmark_with_api.py

# After benchmark completes, run the figure generator
cd ../generatorFigure/
python ./generate_qa_performance_figure.py

echo "QA Benchmark and figure generation complete!" 