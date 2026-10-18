param([Parameter(Mandatory=$true)][string]$LibjpegSource, [Parameter(Mandatory=$true)][string]$LibjpegBuild)
$ErrorActionPreference = 'Stop'
$appRoot = Split-Path $PSScriptRoot
$outputDirectory = Join-Path $appRoot 'runtime/jpeg'
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
gcc -std=c11 -O2 -Wall -Wextra -I (Join-Path $LibjpegSource 'src') -I $LibjpegBuild (Join-Path $PSScriptRoot 'jpeg_local.c') (Join-Path $LibjpegBuild 'libjpeg.a') -static -lbcrypt -lshell32 -o (Join-Path $outputDirectory 'jpeg_local.exe')
if ($LASTEXITCODE) { throw 'JPEG helper compilation failed' }
