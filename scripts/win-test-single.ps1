# Script to run a single pytest test file or test case on Windows
# Usage: .\scripts\win-test-single.ps1 path/to/test.py
# Usage: .\scripts\win-test-single.ps1 "path/to/test.py::test_function"

param(
    [Parameter(Mandatory=$true)]
    [string]$TestPath
)

if ([string]::IsNullOrWhiteSpace($TestPath)) {
    Write-Error "Error: No test path provided"
    Write-Host "Usage: npm run win:test:py:single -- path/to/test.py"
    Write-Host "Usage: npm run win:test:py:single -- `"path/to/test.py::test_function`""
    exit 1
}

# Normalize path to use forward slashes for pytest
$TestPath = $TestPath -replace '\\', '/'

docker compose -f docker-compose.dev.yml run --rm web sh -c "pip install -r requirements.txt -r requirements-dev.txt && python -m pytest -q -n 1 --reuse-db -v $TestPath"
