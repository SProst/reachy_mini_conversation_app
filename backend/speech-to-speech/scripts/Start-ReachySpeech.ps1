[CmdletBinding(PositionalBinding=$false)]
param(
    [switch]$Diarization,
    [string]$Library = $env:NEMO_DIAR_LIBRARY,
    [string]$Model = $env:NEMO_DIAR_MODEL,
    [int]$Gpu = 0,
    [string]$Python = (Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'),
    [string[]]$BackendArgs = @()
)
$ErrorActionPreference = 'Stop'
$names = @('REACHY_DIARIZATION','NEMO_DIAR_LIBRARY','NEMO_DIAR_MODEL','NEMO_DIAR_GPU')
$previous = @{}
foreach ($name in $names) { $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
try {
    $env:REACHY_DIARIZATION = if ($Diarization) { '1' } else { '0' }
    if ($Diarization) {
        if (!(Test-Path -LiteralPath $Library -PathType Leaf) -or !(Test-Path -LiteralPath $Model -PathType Leaf)) {
            throw 'Diarization requires -Library and -Model paths to the separate pinned installation.'
        }
        $env:NEMO_DIAR_LIBRARY = (Resolve-Path -LiteralPath $Library).Path
        $env:NEMO_DIAR_MODEL = (Resolve-Path -LiteralPath $Model).Path
        $env:NEMO_DIAR_GPU = "$Gpu"
    }
    Push-Location (Join-Path $PSScriptRoot '..')
    try {
        # Existing conversation API and ASR/TTS selection; extra serve arguments pass through.
        & $Python -c 'from speech_to_speech.cli import main; main()' serve --stt nemotron-streaming --tts magpie @BackendArgs
        if ($LASTEXITCODE -ne 0) { throw "Speech backend exited with code $LASTEXITCODE" }
    } finally { Pop-Location }
} finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process') }
}
