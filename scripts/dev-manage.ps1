Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$baseArgs = @('compose', '-f', 'docker-compose.yml', '-f', 'docker-compose.dev.yml')
$execArgs = $baseArgs + @('exec')
$runArgs = $baseArgs + @('run','--rm')
if ([Console]::IsInputRedirected) { $execArgs += '-T'; $runArgs += '-T' }

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

# Prefer exec; if it fails (service not running), fallback to run --no-deps
$execFull = $execArgs + $envArgs + @('web','python','manage.py') + $manageArgs
try {
    & docker @execFull
} catch {
    $runFull = $runArgs + @('--no-deps') + $envArgs + @('web','python','manage.py') + $manageArgs
    & docker @runFull
}
