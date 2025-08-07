<# ---------------------------------------------------------------------
   run_gpu.ps1  –  inicializa o stack usando GPU em Windows (Docker Desktop)
   • Detecta GPU e runtime nvidia
   • Instala Toolkit dentro da WSL (Ubuntu/Debian) se faltar
   • Executa docker compose com o override
---------------------------------------------------------------------#>

$ErrorActionPreference = "Stop"

function Log($msg) { Write-Host $msg -ForegroundColor Magenta }

# ─────────────────────────────────────────────────────────────────────────────
# 1) GPU presente?
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Log "⚠️  Nenhuma placa NVIDIA encontrada ou driver ausente. Usando CPU."
    docker compose up --build -d
    exit
}

# ─────────────────────────────────────────────────────────────────────────────
# 2) Runtime nvidia habilitado no Docker?
try {
    $runtimePath = docker info --format "{{json .Runtimes.nvidia}}" 2>$null
} catch { $runtimePath = $null }

if (-not $runtimePath) {
    Log "🔍 GPU detectada, mas runtime NVIDIA não configurado."
    Log "NVIDIA não encontrado. Instalando Toolkit necessário…"

    # detecta a distro WSL padrão
    $distro = & wsl -l --quiet | Select-Object -First 1
    if (-not $distro) {
        Write-Error "Nenhuma distribuição WSL encontrada. Instale uma (ex.: Ubuntu)."
        exit 1
    }

    $cmd = @'
set -e
if ! command -v nvidia-container-toolkit >/dev/null 2>&1; then
  echo "Instalando nvidia-container-toolkit dentro da WSL ($distro)…"
  sudo apt-get update -y
  sudo apt-get install -y curl gnupg lsb-release
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  distribution=$(source /etc/os-release && echo ${ID}${VERSION_ID})
  curl -fsSL "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container.list
  sudo apt-get update -y
  sudo apt-get install -y nvidia-container-toolkit
fi
sudo /etc/init.d/docker restart || true   # reinicia daemon dentro da WSL
'@

    wsl -d $distro -- bash -c "$cmd"
    Log "✅ Toolkit instalado. Reiniciando Docker Desktop…"
    Restart-Service com.docker.service -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 6
}

# ─────────────────────────────────────────────────────────────────────────────
# 3) Sobe com GPU
Log "🚀 Subindo contêiner com suporte a GPU…"
docker compose -f docker-compose.yml -f docker-compose.override.yml up --build -d
Log "✅ Jupyter Lab disponível em http://localhost:4000"
