# Register the checkout's commands for ordinary Windows terminals.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$commandDir = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'bin')).Path
$commands = @('fall-train', 'fall-evaluate', 'fall-detect', 'fall-geometry')

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw 'Create .venv and install the project with pip install -e . first.'
}
# Validate every entry before changing the persistent user PATH.
foreach ($command in $commands) {
    $entry = Join-Path $commandDir ($command + '.cmd')
    & $entry --help | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "$command could not start. Repair the project virtual environment first."
    }
}

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$existingEntries = @($userPath -split ';' | Where-Object { $_.Trim() })
$registered = $existingEntries | Where-Object {
    [Environment]::ExpandEnvironmentVariables($_).TrimEnd('\') -ieq $commandDir.TrimEnd('\')
}
if (-not $registered) {
    $updatedPath = ($existingEntries + $commandDir) -join ';'
    [Environment]::SetEnvironmentVariable('Path', $updatedPath, 'User')
    Write-Output "Added to user PATH: $commandDir"
} else {
    Write-Output "Already registered: $commandDir"
}

# Also expose the commands to child shells launched during this installation.
$processEntries = @($env:Path -split ';')
if (-not ($processEntries | Where-Object { $_.TrimEnd('\') -ieq $commandDir.TrimEnd('\') })) {
    $env:Path = $env:Path.TrimEnd(';') + ';' + $commandDir
}
foreach ($command in $commands) {
    Get-Command $command -CommandType Application | Select-Object Name, Source
}
Write-Output 'Close and reopen your terminal, then run the commands from the project root.'
