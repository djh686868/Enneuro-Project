[CmdletBinding()]
param(
    # The verified GPU is Ada SM 8.9.  Pass -Arch sm_XX when building for a
    # different device; the output DLL name records that architecture.
    [ValidatePattern('^sm_[0-9]+$')]
    [string]$Arch = "sm_89",
    # CUDA 12.6 is the runtime version verified with the EnNeuro CuPy setup.
    [string]$CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
    # Dynamic cudart shares the CUDA Runtime DLL already used by CuPy.  It is
    # the safe default after the static-runtime/CuPy interop smoke check.
    [ValidateSet("shared", "static")]
    [string]$Cudart = "shared"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$sources = Join-Path $root "sources"
$library = Join-Path $sources "library.cu"
$kernels = Join-Path $sources "kernels.cu"
$api = Join-Path $sources "api.h"
$bin = Join-Path $root "bin"
$archTag = $Arch.Replace("_", "")
$dll = Join-Path $bin ("enneuro_cuda_{0}.dll" -f $archTag)
$manifest = Join-Path $bin "build_manifest.json"
$nvcc = Join-Path $CudaPath "bin\nvcc.exe"

foreach ($required in @($library, $kernels, $api, $nvcc)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required build input was not found: $required"
    }
}

$compute = $Arch.Substring(3)
$gencode = "arch=compute_{0},code={1}" -f $compute, $Arch
New-Item -ItemType Directory -Force -Path $bin | Out-Null

# Locate the MSVC x64 environment if this script was not called from a Visual
# Studio developer prompt.  CUDA 12.6 needs cl.exe for the Windows DLL link.
$vsDevCmd = $null
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $installationPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $installationPath) {
            $candidate = Join-Path $installationPath "Common7\Tools\VsDevCmd.bat"
            if (Test-Path -LiteralPath $candidate) { $vsDevCmd = $candidate }
        }
    }
    if (-not $vsDevCmd) {
        throw "MSVC x64 tools were not found. Run from a Visual Studio x64 developer prompt or install VS Build Tools with C++ x64 support."
    }
}

$nvccArguments = @(
    "-O2",
    "--std=c++14",
    "-shared",
    "-Xcompiler", "/MD",
    ("--cudart={0}" -f $Cudart),
    "-gencode", $gencode,
    "-DENEURO_CUDA_BUILD",
    "-I", $sources,
    "-o", $dll,
    $library
)

# Keep a quoted command in the manifest so an experiment can be reproduced.
$quotedNvcc = '"' + $nvcc + '" ' + (($nvccArguments | ForEach-Object {
    if ($_ -match '[\s]') { '"' + $_ + '"' } else { $_ }
}) -join ' ')

if ($vsDevCmd) {
    $command = 'call "' + $vsDevCmd + '" -arch=x64 -host_arch=x64 && ' + $quotedNvcc
    & cmd.exe /d /s /c $command
} else {
    & $nvcc @nvccArguments
    $command = $quotedNvcc
}

if ($LASTEXITCODE -ne 0) {
    throw "nvcc failed with exit code $LASTEXITCODE"
}

$nvccVersion = (& $nvcc --version) -join "`n"
$clVersion = if ($vsDevCmd) { "initialized through $vsDevCmd" } else { ((& cl.exe 2>&1 | Select-Object -First 1) -join "") }
$metadata = [ordered]@{
    abi_version = 1
    architecture = $Arch
    cuda_path = $CudaPath
    cudart = $Cudart
    nvcc_version = $nvccVersion
    msvc = $clVersion
    kernels_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $kernels).Hash
    library_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $library).Hash
    api_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $api).Hash
    command = $command
    built_at = (Get-Date).ToUniversalTime().ToString("o")
}
$metadata | ConvertTo-Json | Set-Content -LiteralPath $manifest -Encoding utf8
Write-Host "Built: $dll"
Write-Host "Manifest: $manifest"
