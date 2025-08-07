#!/usr/bin/env bash
set -e

# ──────────────────────────────────────────────────────────────────────────────
# 0) Função auxiliar para mensagens coloridas
log() { printf '\033[1;35m%s\033[0m\n' "$*"; }

# ──────────────────────────────────────────────────────────────────────────────
# 1) Verifica presença do Toolkit; instala se necessário
if ! command -v nvidia-container-toolkit &>/dev/null; then
  log "⚠️  NVIDIA não encontrado. Instalando Toolkit necessário…"

  if [ -f /etc/os-release ]; then
    . /etc/os-release
    case "$ID" in
      ubuntu|debian)
        log "→ Detectado $PRETTY_NAME (apt)"
        apt-get update -y
        apt-get install -y curl gnupg lsb-release

        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
          gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

        distribution="$(. /etc/os-release; echo ${ID}${VERSION_ID})"
        curl -fsSL "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
          sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
          tee /etc/apt/sources.list.d/nvidia-container.list

        apt-get update -y
        apt-get install -y nvidia-container-toolkit
        ;;
      rhel|centos|fedora)
        log "→ Detectado $PRETTY_NAME (dnf)"
        dnf -y install dnf-plugins-core curl
        dnf config-manager --add-repo \
          "https://nvidia.github.io/libnvidia-container/stable/${ID}/${VERSION_ID}/libnvidia-container.repo"
        dnf install -y nvidia-container-toolkit
        ;;
      *)
        echo "Distribuição $ID não suportada pelo instalador automático."
        echo "Siga instruções oficiais: https://docs.nvidia.com/datacenter/cloud-native/"
        exit 1
        ;;
    esac
  else
    echo "Não foi possível detectar a distribuição Linux. Instale o toolkit manualmente."
    exit 1
  fi

  # Reinicia Docker para carregar o runtime NVIDIA
  systemctl restart docker
  log "✅ NVIDIA Container Toolkit instalado com sucesso."
fi

# ──────────────────────────────────────────────────────────────────────────────
# 2) Sobe a stack com GPU usando overrides
log "🚀 Construindo imagem e iniciando contêiner com suporte a GPU…"
docker compose \
  -f docker-compose.override.yml \
  build 

log "✅ Docker Jupyter Lab construido com sucesso!!!"
