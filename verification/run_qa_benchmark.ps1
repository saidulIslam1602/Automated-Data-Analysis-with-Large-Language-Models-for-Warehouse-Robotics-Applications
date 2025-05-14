# PowerShell script to run QA benchmark

# Change to the benchmark directory
cd .\benchmark\

# Set API key from config.json or from environment
$configPath = "..\data\config.json"
if (Test-Path $configPath) {
    $config = Get-Content $configPath | ConvertFrom-Json
    $env:OPENAI_API_KEY = $config.api_key
}

# Run the benchmark script
python .\qa_benchmark_with_api.py

# After benchmark completes, run the figure generator
cd ..\generatorFigure\
python .\generate_qa_performance_figure.py

Write-Host "QA Benchmark and figure generation complete!" 