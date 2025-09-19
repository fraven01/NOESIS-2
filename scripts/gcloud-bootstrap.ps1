Param(
    [string]$Region,
    [string]$Zone,
    [switch]$FetchSecrets,
    [string[]]$Secrets,
    [string]$OutFile = ".env.gcloud",
    [string]$CiGsa,
    [string]$CiCredFile,
    [switch]$NoCiDetect
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Info($msg) { Write-Host "[i] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[!] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[x] $msg" -ForegroundColor Red }

function Ensure-Command($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Command not found: $name. Please install it and retry."
    }
}

function Select-FromList($Items, $DisplayProp, $ValueProp, $Prompt) {
    if (-not $Items -or $Items.Count -eq 0) { return $null }
    if ($Items.Count -eq 1) { return $Items[0].$ValueProp }
    Write-Host "`n$Prompt" -ForegroundColor Green
    for ($i = 0; $i -lt $Items.Count; $i++) {
        $disp = $Items[$i].$DisplayProp
        Write-Host ("[{0}] {1}" -f ($i+1), $disp)
    }
    while ($true) {
        $sel = Read-Host "Enter number (1-$($Items.Count))"
        if ([int]::TryParse($sel, [ref]$null)) {
            $idx = [int]$sel - 1
            if ($idx -ge 0 -and $idx -lt $Items.Count) { return $Items[$idx].$ValueProp }
        }
        Write-Warn "Invalid selection. Try again."
    }
}

function Get-GcloudJson($args) {
    $json = (& gcloud @args --format=json 2>$null)
    if (-not $json) { return $null }
    try { return $json | ConvertFrom-Json } catch { return $null }
}

try {
    Write-Info "Checking prerequisites (gcloud)"
    Ensure-Command gcloud

    # Auth/account
    $authList = Get-GcloudJson @('auth','list')
    $accounts = @()
    if ($authList) { $accounts = $authList | ForEach-Object { $_.account } | Where-Object { $_ } }
    if (-not $accounts -or $accounts.Count -eq 0) {
        Write-Info "No gcloud account authenticated. Launching login..."
        gcloud auth login | Out-Null
    } else {
        Write-Info ("Found authenticated account(s): {0}" -f ($accounts -join ", "))
    }

    # Project selection
    $projects = Get-GcloudJson @('projects','list')
    if (-not $projects) { throw "No accessible GCP projects via gcloud. Check permissions." }
    $currentProject = (& gcloud config get-value project 2>$null).Trim()
    $defaultProject = $null
    if ($currentProject -and $currentProject -ne '(unset)') { $defaultProject = $currentProject }
    
    $projectChoices = @()
    foreach ($p in $projects) {
        $disp = if ($p.name) { "$($p.projectId) - $($p.name)" } else { $p.projectId }
        $projectChoices += [pscustomobject]@{ display=$disp; value=$p.projectId }
    }
    if ($defaultProject) {
        Write-Info ("Using current gcloud project: {0}" -f $defaultProject)
        $selectedProject = $defaultProject
        $useCurrent = Read-Host "Keep this project? (Y/n)"
        if ($useCurrent -and $useCurrent.Trim().ToLower() -eq 'n') {
            $selectedProject = Select-FromList $projectChoices 'display' 'value' 'Select a GCP project:'
        }
    } else {
        $selectedProject = Select-FromList $projectChoices 'display' 'value' 'Select a GCP project:'
    }
    if (-not $selectedProject) { throw "No project selected." }
    gcloud config set project $selectedProject | Out-Null
    Write-Info ("Project set to: {0}" -f $selectedProject)

    # Region/Zone
    if (-not $Region -or $Region -eq '') {
        $existingRegion = (& gcloud config get-value run/region 2>$null).Trim()
        if (-not $existingRegion -or $existingRegion -eq '(unset)') {
            $existingRegion = (& gcloud config get-value compute/region 2>$null).Trim()
            if (-not $existingRegion -or $existingRegion -eq '(unset)') { $existingRegion = 'europe-west3' }
        }
        $inp = Read-Host ("Preferred region [$existingRegion]")
        $Region = if ($inp) { $inp } else { $existingRegion }
    }
    if ($Region) { gcloud config set compute/region $Region | Out-Null }

    if (-not $Zone -or $Zone -eq '') {
        $existingZone = (& gcloud config get-value compute/zone 2>$null).Trim()
        if (-not $existingZone -or $existingZone -eq '(unset)') { $existingZone = "$Region-a" }
        $inpZ = Read-Host ("Preferred zone [$existingZone]")
        $Zone = if ($inpZ) { $inpZ } else { $existingZone }
    }
    if ($Zone) { gcloud config set compute/zone $Zone | Out-Null }
    Write-Info ("Region/Zone: {0} / {1}" -f $Region, $Zone)

    # Detect CI/GitHub Actions Service Account (optional)
    $ciGsaEmail = $null
    $ciGsaRoles = $null
    if ($CiGsa) { $ciGsaEmail = $CiGsa }
    if (-not $ciGsaEmail -and $CiCredFile -and (Test-Path $CiCredFile)) {
        try {
            $json = Get-Content -Raw -ErrorAction Stop $CiCredFile | ConvertFrom-Json
            if ($json.client_email) { $ciGsaEmail = $json.client_email }
        } catch { Write-Warn ("Could not parse CI credentials file: {0}" -f $_.Exception.Message) }
    }
    if (-not $ciGsaEmail -and -not $NoCiDetect) {
        # Heuristic 1: service accounts list with CI-like names
        $saList = (& gcloud iam service-accounts list --format=value(email) 2>$null) -split "`n" | Where-Object { $_ }
        foreach ($sa in $saList) {
            if ($sa -match '(github|actions|ci|cd|deploy|pipeline)') { $ciGsaEmail = $sa; break }
        }
        # Heuristic 2: IAM policy roles
        if (-not $ciGsaEmail) {
            $iamRows = (& gcloud projects get-iam-policy $selectedProject --flatten="bindings[].members" --filter="bindings.members:serviceAccount:" --format='csv[no-heading](bindings.role,bindings.members)' 2>$null) -split "`n" | Where-Object { $_ }
            $desired = 'roles/run\.admin|roles/run\.developer|roles/artifactregistry\.writer|roles/artifactregistry\.admin|roles/cloudsql\.client|roles/iam\.serviceAccountUser|roles/container\.admin|roles/container\.developer'
            $roleCountBySa = @{}
            $rolesBySa = @{}
            foreach ($row in $iamRows) {
                $parts = $row.Split(',',2)
                if ($parts.Count -lt 2) { continue }
                $role = $parts[0].Trim()
                $member = $parts[1].Trim()
                if ($member -like 'serviceAccount:*') {
                    $saEmail = $member.Substring('serviceAccount:'.Length)
                    if ($role -match $desired) {
                        if (-not $roleCountBySa.ContainsKey($saEmail)) { $roleCountBySa[$saEmail] = 0; $rolesBySa[$saEmail] = @() }
                        $roleCountBySa[$saEmail] = $roleCountBySa[$saEmail] + 1
                        $rolesBySa[$saEmail] += $role
                    }
                }
            }
            $bestSa = $null; $bestCount = -1
            foreach ($sa in $roleCountBySa.Keys) {
                if ($sa -match '(github|actions|ci|cd|deploy|pipeline)') { $ciGsaEmail = $sa; $ciGsaRoles = ($rolesBySa[$sa] | Select-Object -Unique) -join ','; break }
                $count = [int]$roleCountBySa[$sa]
                if ($count -gt $bestCount) { $bestCount = $count; $bestSa = $sa }
            }
            if (-not $ciGsaEmail -and $bestSa) { $ciGsaEmail = $bestSa; $ciGsaRoles = ($rolesBySa[$bestSa] | Select-Object -Unique) -join ',' }
        }
    }

    # Discover Redis (MemoryStore)
    $redisInstances = Get-GcloudJson @('redis','instances','list','--region', $Region)
    $redisHost = ''
    $redisPort = ''
    $redisName = ''
    if ($redisInstances -and $redisInstances.Count -gt 0) {
        $choices = @()
        foreach ($r in $redisInstances) {
            $disp = "$($r.name) - $($r.host):$($r.port)"
            $choices += [pscustomobject]@{ display=$disp; value=$r.name }
        }
        $selRedis = Select-FromList $choices 'display' 'value' "Select Redis instance in $Region (Memorystore):"
        if ($selRedis) {
            $ri = $redisInstances | Where-Object { $_.name -eq $selRedis } | Select-Object -First 1
            if ($ri) { $redisName = $ri.name; $redisHost = $ri.host; $redisPort = $ri.port }
        }
    } else {
        Write-Warn "No Redis instances found in region $Region"
    }

    # Discover Cloud SQL (Postgres)
    $sqlInstances = Get-GcloudJson @('sql','instances','list')
    $pgInstances = @()
    foreach ($s in ($sqlInstances | Where-Object { $_.databaseVersion -like 'POSTGRES*' })) {
        $ip = $null; if ($s.ipAddresses -and $s.ipAddresses.Count -gt 0) { $ip = $s.ipAddresses[0].ipAddress }
        $disp = "$($s.name) - $($s.region) - IP: $ip"
        $pgInstances += [pscustomobject]@{ display=$disp; value=$s.name }
    }
    $sqlName = ''
    $sqlRegion = ''
    $sqlConn = ''
    $sqlIp = ''
    if ($pgInstances -and $pgInstances.Count -gt 0) {
        $selSql = Select-FromList $pgInstances 'display' 'value' 'Select Cloud SQL (Postgres) instance:'
        if ($selSql) {
            $si = $sqlInstances | Where-Object { $_.name -eq $selSql } | Select-Object -First 1
            if ($si) {
                $sqlName = $si.name
                $sqlRegion = $si.region
                $sqlConn = $si.connectionName
                if ($si.ipAddresses -and $si.ipAddresses.Count -gt 0) { $sqlIp = $si.ipAddresses[0].ipAddress }
            }
        }
    } else {
        Write-Warn "No Cloud SQL Postgres instances found"
    }

    # Discover Cloud Run services
    $runServices = Get-GcloudJson @('run','services','list','--platform','managed','--region',$Region)
    $runPairs = @()
    if ($runServices) {
        foreach ($svc in $runServices) {
            $runPairs += [pscustomobject]@{ name=$svc.metadata.name; url=$svc.status.url }
        }
    }

    # Discover Artifact Registry (docker) in region
    $repos = Get-GcloudJson @('artifacts','repositories','list','--location',$Region)
    $dockerRepos = @()
    if ($repos) {
        foreach ($r in ($repos | Where-Object { $_.format -eq 'DOCKER' })) {
            # repo resource name looks like: projects/PROJECT/locations/REGION/repositories/REPO
            $repoName = ($r.name -split '/repositories/')[1]
            $dockerRepos += [pscustomobject]@{ repo=$repoName; location=$r.location; host=("{0}-docker.pkg.dev" -f $r.location) }
        }
    }

    # Build .env.gcloud content
    $lines = @()
    $lines += "# Generated by scripts/gcloud-bootstrap.ps1 on $(Get-Date -Format o)"
    $lines += "# Do not commit this file. Contains environment hints from GCP."
    $lines += "# Project/Region"
    $lines += "GCP_PROJECT=$selectedProject"
    $lines += "GCP_REGION=$Region"
    $lines += "GCP_ZONE=$Zone"
    $lines += ""
    if ($redisHost) {
        $lines += "# Redis (Memorystore)"
        $lines += "GCP_REDIS_INSTANCE=$redisName"
        $lines += "GCP_REDIS_HOST=$redisHost"
        $lines += "GCP_REDIS_PORT=$redisPort"
        $lines += "GCP_REDIS_URL=redis://$($redisHost):$($redisPort)/0"
        $lines += ""
    }
    if ($ciGsaEmail) {
        $lines += "# GitHub Actions / CI Service Account (detected or provided)"
        $lines += "GITHUB_ACTIONS_GSA_EMAIL=$ciGsaEmail"
        if ($ciGsaRoles) { $lines += "GITHUB_ACTIONS_GSA_ROLES=$ciGsaRoles" }
        $lines += ""
    }
    if ($sqlName) {
        $lines += "# Cloud SQL (Postgres)"
        $lines += "GCP_SQL_INSTANCE=$sqlName"
        $lines += "GCP_SQL_REGION=$sqlRegion"
        if ($sqlIp) { $lines += "GCP_SQL_PUBLIC_IP=$sqlIp" }
        if ($sqlConn) { $lines += "GCP_SQL_CONNECTION_NAME=$sqlConn" }
        $lines += "# For local dev, keep DB_USER/DB_PASSWORD from .env; optionally point DB_HOST to GCP_SQL_PUBLIC_IP"
        $lines += ""
    }
    if ($runPairs -and $runPairs.Count -gt 0) {
        $lines += "# Cloud Run services in $Region"
        foreach ($p in $runPairs) {
            $key = ($p.name.ToUpper() -replace "[^A-Z0-9]","_")
            $lines += ("GCP_RUN_{0}_URL={1}" -f $key, $p.url)
        }
        $lines += ""
    }
    if ($dockerRepos -and $dockerRepos.Count -gt 0) {
        $lines += "# Artifact Registry docker repos in $Region"
        foreach ($dr in $dockerRepos) {
            $lines += ("GCP_ARTIFACT_REGISTRY_HOST={0}" -f $dr.host)
            $lines += ("GCP_ARTIFACT_REGISTRY_REPO={0}" -f $dr.repo)
        }
        $lines += ""
    }

    if (Test-Path $OutFile) {
        Write-Warn "$OutFile already exists. Overwrite? (y/N)"
        $ans = Read-Host
        if ($ans.ToLower() -ne 'y') { Write-Err "Aborted to avoid overwrite."; exit 1 }
    }
    $lines | Set-Content -NoNewline:$false -Encoding UTF8 $OutFile
    Write-Info ("Wrote {0}" -f $OutFile)

    if ($FetchSecrets) {
        if (-not $Secrets -or $Secrets.Count -eq 0) {
            Write-Warn "-FetchSecrets was set but no -Secrets provided. Skipping secrets fetch."
        } else {
            Write-Warn "Fetching secret payloads will write sensitive values to $OutFile. Handle with care!"
            foreach ($s in $Secrets) {
                try {
                    $val = (& gcloud secrets versions access latest --secret=$s 2>$null)
                    if ($null -ne $val -and $val.Trim() -ne '') {
                        Add-Content -Encoding UTF8 -Path $OutFile -Value ("{0}={1}" -f $s.Trim(), $val.Trim())
                        Write-Info ("Wrote secret: {0}" -f $s)
                    } else {
                        Write-Warn ("Empty or unavailable secret: {0}" -f $s)
                    }
                } catch {
                    Write-Warn ("Failed to access secret {0}: {1}" -f $s, $_.Exception.Message)
                }
            }
        }
    }

    Write-Host "`nDone. Suggested next steps:" -ForegroundColor Green
    Write-Host "  - Review $OutFile and selectively copy values into your .env as needed."
    Write-Host "  - Do not commit $OutFile."
    Write-Host "  - For Cloud SQL, prefer local Postgres for dev or use Cloud SQL Auth Proxy."

} catch {
    Write-Err $_.Exception.Message
    exit 1
}
