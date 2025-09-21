Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$dockerArgs = @('compose', '-f', 'docker-compose.yml', '-f', 'docker-compose.dev.yml', 'exec')
if ([Console]::IsInputRedirected) {
    $dockerArgs += '-T'
}

$envArgs = @()
$index = 0
while ($index -lt $args.Count) {
    $current = $args[$index]
    if ($current -eq '--') {
        $index++
        break
    } elseif ($current -eq '--env') {
        if ($index + 1 -ge $args.Count) {
            throw '--env erwartet KEY=VALUE'
        }
        $envArgs += @('-e', $args[$index + 1])
        $index += 2
        continue
    } elseif ($current -like '*=*') {
        $envArgs += @('-e', $current)
        $index++
        continue
    } else {
        break
    }
}

if ($index -ge $args.Count) {
    Write-Error "Usage: dev-manage.ps1 [VAR=value ...] <command> [args...]"
    exit 1
}

$manageArgs = @()
for ($j = $index; $j -lt $args.Count; $j++) {
    $manageArgs += $args[$j]
}

$fullArgs = $dockerArgs + $envArgs + @('web', 'python', 'manage.py') + $manageArgs

& docker @fullArgs
