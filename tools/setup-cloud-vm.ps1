<#
.SYNOPSIS
  Sets up Skyforge on a fresh Windows cloud VM with an NVIDIA GPU.

.DESCRIPTION
  Suggested VMs: Azure Standard_NC8as_T4_v3 (56 GB RAM) or AWS g4dn.2xlarge
  (32 GB RAM), Windows Server 2022 or 2025, about 256 GB of disk.

  1. On your laptop, from the project folder, package the code. This includes
     uncommitted and untracked files, which a git clone would miss:

       powershell -ExecutionPolicy Bypass -File tools\setup-cloud-vm.ps1 -Pack

  2. Create the VM. Allow Remote Desktop (port 3389) from your own IP only.
     Do NOT open port 8787: the Skyforge server has no login.

  3. On the VM, install the NVIDIA driver (this script does not):
       Azure: add the "NVIDIA GPU Driver Extension" to the VM
              https://learn.microsoft.com/azure/virtual-machines/extensions/hpccompute-gpu-windows
       AWS:   https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html
     Reboot, and check that nvidia-smi runs.

  4. Copy tmp\skyforge-source.zip to the VM (paste it into the Remote Desktop
     window), extract it to C:\skyforge, open PowerShell as Administrator there:

       powershell -ExecutionPolicy Bypass -File tools\setup-cloud-vm.ps1

  5. Run "npm start" and open http://localhost:8787 in the VM's browser.

  6. Download the results, then stop the VM from the cloud portal. On Azure use
     "Stop (deallocate)"; shutting down inside Windows keeps charging for compute.
     Disks are billed until you delete them.

  The script can be run again after a failure: finished steps are skipped.
  It installs the Microsoft Visual C++ runtime, Node.js, Python 3.13, FFmpeg,
  COLMAP, OpenMVS, the Python packages and the npm packages, and adds a Windows
  Firewall rule that blocks incoming connections to port 8787.

.PARAMETER Pack
  Laptop side: zip the project source into tmp\skyforge-source.zip and exit.

.PARAMETER SkipSplat
  Do not install the Gaussian splat environment (about 3 GB).

.PARAMETER CheckOnly
  Install nothing; only report whether this machine is ready.
#>
[CmdletBinding()]
param(
  [switch]$Pack,
  [switch]$SkipSplat,
  [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # the progress bar makes downloads many times slower
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$Root = Split-Path -Parent $PSScriptRoot
$Port = 8787
$NodeVersion = '22.22.1'
$PythonVersion = '3.13.12'

$Urls = @{
  VcRedist = 'https://aka.ms/vc14/vc_redist.x64.exe'
  Node     = "https://nodejs.org/dist/v$NodeVersion/node-v$NodeVersion-x64.msi"
  NodeSums = "https://nodejs.org/dist/v$NodeVersion/SHASUMS256.txt"
  Python   = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
  Ffmpeg   = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
  Colmap   = 'https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip'
  OpenMvs  = 'https://github.com/cdcseacave/openMVS/releases/download/v2.3.0/OpenMVS_Windows_x64.7z'
  SevenZip = 'https://www.7-zip.org/a/7zr.exe'
}
# From engines/README.md: the OpenMVS archive this project was verified with.
$OpenMvsSha256 = 'E1B7AE31AD078160059005B1BFED10280C14E99EE2579F5F806CCAFF05C784CA'

function Write-Step([string]$Text) { Write-Host "`n== $Text" -ForegroundColor Cyan }

function Invoke-Checked([string]$What, [scriptblock]$Command) {
  & $Command
  if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE." }
}

# cmd.exe merges stderr into stdout; doing that inside PowerShell 5.1 raises errors.
function Get-NativeOutput([string]$CommandLine) {
  return ((cmd.exe /d /c "$CommandLine 2>&1") -join "`n")
}

function Update-SessionPath {
  $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
  $user = [Environment]::GetEnvironmentVariable('Path', 'User')
  $env:Path = "$machine;$user"
}

function Save-Download([string]$Url, [string]$FileName) {
  $target = Join-Path $script:Downloads $FileName
  Write-Host "Downloading $Url"
  Invoke-WebRequest -Uri $Url -OutFile $target -UseBasicParsing
  return $target
}

function Assert-Sha256([string]$Path, [string]$Expected) {
  $actual = (Get-FileHash -Path $Path -Algorithm SHA256).Hash
  if ($actual -ne $Expected.Trim().ToUpperInvariant()) {
    Remove-Item $Path -Force
    throw "Checksum mismatch for $Path (expected $Expected, got $actual). The file was deleted."
  }
}

function Assert-Signature([string]$Path, [string]$Publisher) {
  $signature = Get-AuthenticodeSignature -FilePath $Path
  if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch $Publisher) {
    throw "$Path is not validly signed by $Publisher (status: $($signature.Status))."
  }
}

function Install-Package([string]$File, [string]$Arguments) {
  $process = Start-Process -FilePath $File -ArgumentList $Arguments -Wait -PassThru
  # 1638: a newer version is already installed. 3010: success, reboot wanted.
  if (@(0, 1638, 3010) -notcontains $process.ExitCode) { throw "$File exited with code $($process.ExitCode)." }
  if ($process.ExitCode -eq 3010) { $script:RebootWanted = $true }
  Update-SessionPath
}

function Expand-ToTemp([string]$Zip, [string]$Name) {
  $folder = Join-Path $script:Downloads $Name
  if (Test-Path $folder) { Remove-Item $folder -Recurse -Force }
  Expand-Archive -Path $Zip -DestinationPath $folder
  return $folder
}

function Find-NvidiaSmi {
  $found = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
  if ($found) { return $found.Source }
  $legacy = Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe'
  if (Test-Path $legacy) { return $legacy }
  return $null
}

# ---------------------------------------------------------------- laptop side
function New-SourcePackage {
  if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'git is needed to list the project files.' }
  Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
  $outDir = Join-Path $Root 'tmp'
  New-Item -ItemType Directory -Force $outDir | Out-Null
  $zipPath = Join-Path $outDir 'skyforge-source.zip'
  if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

  Push-Location $Root
  try {
    # Tracked plus untracked files, minus everything .gitignore excludes.
    $files = git -c core.quotepath=off ls-files --cached --others --exclude-standard
    if ($LASTEXITCODE -ne 0) { throw 'git ls-files failed.' }
  } finally { Pop-Location }

  $zip = [IO.Compression.ZipFile]::Open($zipPath, 'Create')
  $count = 0
  try {
    foreach ($relative in $files) {
      if ($relative -like '.claude/*') { continue }
      $full = Join-Path $Root $relative
      if (-not (Test-Path -LiteralPath $full -PathType Leaf)) { continue }
      [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $full, $relative) | Out-Null
      $count++
    }
  } finally { $zip.Dispose() }

  $sizeMb = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
  Write-Host "Packed $count files ($sizeMb MB) into $zipPath"
  Write-Host 'Videos, runtime/, output/ and the engines are not included.'
}

# --------------------------------------------------------------- verification
function Test-Setup {
  Write-Step 'Checking the installation'
  $results = @()
  function Add-Result($Name, $Ok, $Detail, $Required = $true) {
    [pscustomobject]@{ Check = $Name; Ok = [bool]$Ok; Required = $Required; Detail = "$Detail" }
  }

  $smi = Find-NvidiaSmi
  if ($smi) {
    $gpu = Get-NativeOutput "`"$smi`" --query-gpu=name,memory.total,driver_version --format=csv,noheader"
    $results += Add-Result 'NVIDIA GPU' ($LASTEXITCODE -eq 0) $gpu
  } else {
    $results += Add-Result 'NVIDIA GPU' $false 'nvidia-smi not found; install the driver'
  }

  $ramGb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
  $results += Add-Result 'RAM of 24 GB or more' ($ramGb -ge 23.5) "$ramGb GB" $false

  $node = Get-Command node -ErrorAction SilentlyContinue
  $nodeText = if ($node) { Get-NativeOutput 'node --version' } else { 'not found' }
  $results += Add-Result 'Node 22 or newer' ($nodeText -match '^v(\d+)' -and [int]$Matches[1] -ge 22) $nodeText

  $pythonText = if (Get-Command python -ErrorAction SilentlyContinue) { Get-NativeOutput 'python --version' } else { 'not found' }
  $results += Add-Result 'Python 3.13' ($pythonText -match 'Python 3\.13\.') $pythonText

  $ffmpeg = Join-Path $Root 'engines\ffmpeg\bin\ffmpeg.exe'
  if (Test-Path $ffmpeg) {
    $version = (Get-NativeOutput "`"$ffmpeg`" -hide_banner -version") -split "`n" | Select-Object -First 1
    $results += Add-Result 'FFmpeg' ($version -match 'ffmpeg version') $version
    $filters = Get-NativeOutput "`"$ffmpeg`" -hide_banner -filters"
    $results += Add-Result 'FFmpeg GPU resize' ($filters -match 'scale_cuda') 'scale_cuda; without it, frames are decoded on the CPU' $false
  } else {
    $results += Add-Result 'FFmpeg' $false 'missing'
  }

  $colmap = Join-Path $Root 'engines\colmap\bin\colmap.exe'
  if (Test-Path $colmap) {
    $help = Get-NativeOutput "`"$colmap`" help"
    $line = $help -split "`n" | Where-Object { $_ -match '^COLMAP ' } | Select-Object -First 1
    $results += Add-Result 'COLMAP with CUDA' ($help -match 'with CUDA') $line
  } else {
    $results += Add-Result 'COLMAP with CUDA' $false 'missing'
  }

  $mvsMissing = @('InterfaceCOLMAP', 'DensifyPointCloud', 'ReconstructMesh', 'RefineMesh', 'TextureMesh') |
    Where-Object { -not (Test-Path (Join-Path $Root "engines\openmvs\$_.exe")) }
  $results += Add-Result 'OpenMVS files' (-not $mvsMissing) $(if ($mvsMissing) { "missing: $($mvsMissing -join ', ')" } else { 'present' })

  $cudnn = Join-Path $Root 'engines\cudnn\nvidia\cudnn\bin\cudnn64_9.dll'
  $results += Add-Result 'cuDNN 9' (Test-Path $cudnn) 'learned features (ALIKED + LightGlue)'

  $results += Add-Result 'npm packages' (Test-Path (Join-Path $Root 'node_modules\vite')) 'node_modules'
  $results += Add-Result 'Web UI built' (Test-Path (Join-Path $Root 'dist\index.html')) 'dist/'

  if (-not $SkipSplat) {
    $venv = Join-Path $Root '.venv-splat\Scripts\python.exe'
    if (Test-Path $venv) {
      $probe = Get-NativeOutput "`"$venv`" -c `"import torch, gsplat; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))`""
      $results += Add-Result 'Gaussian splat (torch sees GPU)' ($LASTEXITCODE -eq 0) ($probe -split "`n" | Select-Object -Last 1)
    } else {
      $results += Add-Result 'Gaussian splat (torch sees GPU)' $false 'not installed'
    }
  }

  $rule = Get-NetFirewallRule -DisplayName "Skyforge: block inbound $Port" -ErrorAction SilentlyContinue
  $results += Add-Result "Port $Port blocked from outside" $rule 'Windows Firewall rule' $false

  foreach ($r in $results) {
    $mark = if ($r.Ok) { 'OK  ' } elseif ($r.Required) { 'FAIL' } else { 'WARN' }
    $color = if ($r.Ok) { 'Green' } elseif ($r.Required) { 'Red' } else { 'Yellow' }
    Write-Host ("{0}  {1,-34} {2}" -f $mark, $r.Check, $r.Detail) -ForegroundColor $color
  }
  return @($results | Where-Object { $_.Required -and -not $_.Ok }).Count
}

# ----------------------------------------------------------------------- main
if ($Pack) { New-SourcePackage; return }

if (-not (Test-Path (Join-Path $Root 'package.json'))) { throw "package.json not found in $Root. Run this from inside the project." }
if (-not [Environment]::Is64BitOperatingSystem) { throw '64-bit Windows is required.' }

if ($CheckOnly) {
  $failed = Test-Setup
  if ($failed) { Write-Host "`n$failed required check(s) failed."; exit 1 }
  Write-Host "`nAll required checks passed."
  exit 0
}

$identity = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $identity.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  throw 'Open PowerShell with "Run as administrator" and run the script again.'
}

Write-Step 'Checking the GPU driver'
if (-not (Find-NvidiaSmi)) {
  throw 'nvidia-smi not found. Install the NVIDIA driver (step 3 in this script''s help), reboot, and run again.'
}
Invoke-Checked 'nvidia-smi' { & (Find-NvidiaSmi) '--query-gpu=name,memory.total,driver_version' '--format=csv,noheader' }

$script:Downloads = Join-Path $env:TEMP 'skyforge-setup'
New-Item -ItemType Directory -Force $script:Downloads | Out-Null
$script:RebootWanted = $false

Write-Step "Blocking incoming connections to port $Port"
$ruleName = "Skyforge: block inbound $Port"
if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
  # A block rule does not affect http://localhost, so the app still works on the VM.
  New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Block -Protocol TCP -LocalPort $Port | Out-Null
}

Write-Step 'Microsoft Visual C++ runtime'
$vc = Save-Download $Urls.VcRedist 'vc_redist.x64.exe'
Assert-Signature $vc 'Microsoft Corporation'
Install-Package $vc '/install /quiet /norestart'

Write-Step "Node.js $NodeVersion"
Update-SessionPath
$haveNode = (Get-Command node -ErrorAction SilentlyContinue) -and ((Get-NativeOutput 'node --version') -match '^v(\d+)') -and ([int]$Matches[1] -ge 22)
if ($haveNode) { Write-Host 'Already installed.' } else {
  $msi = Save-Download $Urls.Node "node-v$NodeVersion-x64.msi"
  $sums = Save-Download $Urls.NodeSums 'node-SHASUMS256.txt'
  $line = Get-Content $sums | Where-Object { $_ -match "\snode-v$([regex]::Escape($NodeVersion))-x64\.msi$" } | Select-Object -First 1
  if (-not $line) { throw 'Node checksum list does not include the installer.' }
  Assert-Sha256 $msi ($line -split '\s+')[0]
  Install-Package 'msiexec.exe' "/i `"$msi`" /qn /norestart"
}

Write-Step "Python $PythonVersion"
$havePython = (Get-Command python -ErrorAction SilentlyContinue) -and ((Get-NativeOutput 'python --version') -match 'Python 3\.13\.')
if ($havePython) { Write-Host 'Already installed.' } else {
  $installer = Save-Download $Urls.Python "python-$PythonVersion-amd64.exe"
  Assert-Signature $installer 'Python Software Foundation'
  Install-Package $installer '/quiet InstallAllUsers=1 PrependPath=1 Include_launcher=1 Include_test=0'
  if ((Get-NativeOutput 'python --version') -notmatch 'Python 3\.13\.') {
    throw '"python" does not resolve to 3.13. Remove the other Python from PATH and run again.'
  }
}

Write-Step 'FFmpeg'
$ffmpegBin = Join-Path $Root 'engines\ffmpeg\bin'
if ((Test-Path "$ffmpegBin\ffmpeg.exe") -and (Test-Path "$ffmpegBin\ffprobe.exe")) { Write-Host 'Already installed.' } else {
  $zip = Save-Download $Urls.Ffmpeg 'ffmpeg.zip'
  $sumFile = Save-Download "$($Urls.Ffmpeg).sha256" 'ffmpeg.zip.sha256'
  Assert-Sha256 $zip ((Get-Content $sumFile -Raw).Trim() -split '\s+')[0]
  $folder = Expand-ToTemp $zip 'ffmpeg'
  New-Item -ItemType Directory -Force $ffmpegBin | Out-Null
  foreach ($name in 'ffprobe.exe', 'ffmpeg.exe') {
    $source = Get-ChildItem $folder -Recurse -Filter $name | Select-Object -First 1
    if (-not $source) { throw "$name not found in the FFmpeg archive." }
    Copy-Item $source.FullName $ffmpegBin -Force
  }
}

Write-Step 'COLMAP 4.2.1 (CUDA, about 400 MB)'
$colmapDir = Join-Path $Root 'engines\colmap'
if (Test-Path "$colmapDir\bin\colmap.exe") { Write-Host 'Already installed.' } else {
  # No pinned checksum for this archive; it comes over HTTPS from the official release.
  $zip = Save-Download $Urls.Colmap 'colmap.zip'
  $folder = Expand-ToTemp $zip 'colmap'
  $exe = Get-ChildItem $folder -Recurse -Filter 'colmap.exe' | Select-Object -First 1
  if (-not $exe) { throw 'colmap.exe not found in the COLMAP archive.' }
  New-Item -ItemType Directory -Force $colmapDir | Out-Null
  Copy-Item (Join-Path $exe.Directory.Parent.FullName '*') $colmapDir -Recurse -Force
}

Write-Step 'OpenMVS 2.3.0'
$mvsDir = Join-Path $Root 'engines\openmvs'
if ((Test-Path "$mvsDir\DensifyPointCloud.exe") -and (Test-Path "$mvsDir\TextureMesh.exe")) { Write-Host 'Already installed.' } else {
  $archive = Save-Download $Urls.OpenMvs 'openmvs.7z'
  Assert-Sha256 $archive $OpenMvsSha256
  # 7zr, because the tar shipped with some Windows Server versions cannot read .7z.
  $sevenZip = Save-Download $Urls.SevenZip '7zr.exe'
  $folder = Join-Path $script:Downloads 'openmvs'
  if (Test-Path $folder) { Remove-Item $folder -Recurse -Force }
  Invoke-Checked 'Extracting OpenMVS' { & $sevenZip x $archive "-o$folder" -y | Out-Null }
  $densify = Get-ChildItem $folder -Recurse -Filter 'DensifyPointCloud.exe' | Select-Object -First 1
  if (-not $densify) { throw 'DensifyPointCloud.exe not found in the OpenMVS archive.' }
  New-Item -ItemType Directory -Force $mvsDir | Out-Null
  Get-ChildItem $densify.Directory.FullName -File | Where-Object { $_.Extension -in '.exe', '.dll' } |
    Copy-Item -Destination $mvsDir -Force
}

Push-Location $Root
try {
  Write-Step 'npm packages'
  Invoke-Checked 'npm ci' { npm.cmd ci --no-audit --no-fund }

  Write-Step 'Python packages for sky masking'
  Invoke-Checked 'pip install' { python -m pip install --no-cache-dir -r tools\python\requirements-mask.txt }

  Write-Step 'cuDNN 9 (about 2 GB)'
  if (Test-Path 'engines\cudnn\nvidia\cudnn\bin\cudnn64_9.dll') { Write-Host 'Already installed.' } else {
    Invoke-Checked 'npm run setup:cudnn' { npm.cmd run setup:cudnn }
  }

  if ($SkipSplat) { Write-Host "`nSkipping Gaussian splatting (-SkipSplat)." } else {
    Write-Step 'Gaussian splat environment (about 3 GB)'
    $venv = '.venv-splat\Scripts\python.exe'
    $ready = $false
    if (Test-Path $venv) {
      Get-NativeOutput "$venv -c `"import torch, gsplat`"" | Out-Null
      $ready = $LASTEXITCODE -eq 0
    }
    if ($ready) { Write-Host 'Already installed.' } else {
      Invoke-Checked 'npm run setup:splat' { npm.cmd run setup:splat }
    }
  }

  Write-Step 'Building the web UI'
  Invoke-Checked 'npm run build' { npm.cmd run build }
} finally { Pop-Location }

$failed = Test-Setup
Remove-Item $script:Downloads -Recurse -Force -ErrorAction SilentlyContinue

if ($script:RebootWanted) { Write-Host "`nAn installer asked for a reboot. Reboot before the first run." -ForegroundColor Yellow }
if ($failed) {
  Write-Host "`n$failed required check(s) failed. Fix them and run the script again." -ForegroundColor Red
  exit 1
}
Write-Host @"

Setup finished. Start the app from a new PowerShell window:

  cd "$Root"
  npm start

Then open http://localhost:$Port in this VM's browser.
These checks do not exercise OpenMVS on this GPU, so run one short video with
the fast profile before the full multi-video job.
"@ -ForegroundColor Green
