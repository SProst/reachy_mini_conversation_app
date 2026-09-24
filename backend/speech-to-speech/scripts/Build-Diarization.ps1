[CmdletBinding()]
param(
    [string]$CMake = (Join-Path $PSScriptRoot '../.venv/Scripts/cmake.exe'),
    [string]$Python = (Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'),
    [switch]$DownloadModel
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
function Invoke-Checked([string]$Binary, [string[]]$Arguments) {
    & $Binary @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Binary exited with code $LASTEXITCODE" }
}
function Get-PinnedSource([string]$Directory, [string]$Url, [string]$Revision) {
    if (!(Test-Path -LiteralPath $Directory)) {
        Invoke-Checked git @('clone','--filter=blob:none','--no-checkout',$Url,$Directory)
        Invoke-Checked git @('-C',$Directory,'checkout',$Revision)
    }
    $head = & git -C $Directory rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $head -ne $Revision) { throw "Existing source at $Directory does not match pinned revision $Revision" }
}
Push-Location $root
try {
    $source = Join-Path $root '.diar-runtime'
    Get-PinnedSource $source 'https://github.com/NVIDIA/NeMo-Speech.cpp.git' '97a15afa5caa9bce5baaa86c1184103877af4101'
    Invoke-Checked git @('-C',$source,'submodule','update','--init','--depth','1','ggml')
    $spmSource = Join-Path $source '.deps/sentencepiece-source'
    $spmBuild = Join-Path $source '.deps/sentencepiece-build'
    Get-PinnedSource $spmSource 'https://github.com/google/sentencepiece.git' '17d7580d6407802f85855d2cc9190634e2c95624'
    Invoke-Checked $CMake @('-S',$spmSource,'-B',$spmBuild,'-G','Visual Studio 17 2022','-A','x64',
        '-DSPM_BUILD_TEST=OFF','-DSPM_ENABLE_SHARED=OFF','-DSPM_ENABLE_TCMALLOC=OFF','-DCMAKE_POLICY_VERSION_MINIMUM=3.5')
    Invoke-Checked $CMake @('--build',$spmBuild,'--config','Release','--target','sentencepiece-static','--parallel','4')
    $build = Join-Path $source 'build/windows-cpu'
    Invoke-Checked $CMake @('-S',$source,'-B',$build,'-G','Visual Studio 17 2022','-A','x64',
        '-DNEMO_SPEECH_BUILD_ASR=OFF','-DNEMO_SPEECH_BUILD_DIAR=ON','-DNEMO_SPEECH_BUILD_TTS=OFF',
        '-DNEMO_SPEECH_BUILD_NMT=OFF','-DNEMO_SPEECH_BUILD_CLI=OFF','-DNEMO_SPEECH_BUILD_MIC_CAPTURE=OFF',
        '-DGGML_NATIVE=OFF','-DNEMO_SPEECH_GGML_PATCHED=OFF',
        "-DSENTENCEPIECE_LIB=$spmBuild/src/Release/sentencepiece.lib","-DSENTENCEPIECE_INCLUDE_DIR=$spmSource/src")
    Invoke-Checked $CMake @('--build',$build,'--config','Release','--parallel','4')
    Invoke-Checked $CMake @('--install',$build,'--config','Release','--prefix',(Join-Path $root '.diar-install'))
    if ($DownloadModel) {
        Invoke-Checked $Python @('-c', "from huggingface_hub import hf_hub_download; hf_hub_download('nvidia/Nemotron-3-Diarization','Nemotron-3-Diarization.q8_0.gguf',revision='f667ed73aee57d40cc39428eb768b4fd87a0a29e',local_dir='.diar-models')")
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath '.diar-models/Nemotron-3-Diarization.q8_0.gguf').Hash
        if ($hash -ne '08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1') { throw 'Model checksum mismatch' }
    }
    Write-Output 'CPU diarization installed independently in .diar-install. Use -Gpu -1 in Start-ReachySpeech.ps1.'
} finally { Pop-Location }
