#!/usr/bin/env bash
set -Eeuo pipefail

LOCAL_DIR="/mnt/c/Users/32878/Desktop/lt/personal-assistant-agent"
REMOTE_HOST="myserver"
REMOTE_DIR="/home/admin/projects/personal-assistant-agent"
BACKUP_DIR="/home/admin/deploy-backups/personal-assistant"
BACKUP_ARCHIVE="$BACKUP_DIR/source-rollback.tar.gz"

CURRENT_IMAGE="personal-assistant-agent:dev"
ROLLBACK_IMAGE="personal-assistant-agent:rollback"

echo "[1/4] 保存服务器当前版本"
ssh "$REMOTE_HOST" "
  set -e
  mkdir -p '$BACKUP_DIR'
  docker image inspect '$CURRENT_IMAGE' >/dev/null
  docker image tag '$CURRENT_IMAGE' '$ROLLBACK_IMAGE'
  tar \
    --exclude='.env' \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='.idea' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='assistant.db' \
    --exclude='chroma_data' \
    -czf '$BACKUP_ARCHIVE' \
    -C '$REMOTE_DIR' .
"

echo "[2/4] 同步本地代码"
rsync -av --delete \
  --exclude='.git/' \
  --exclude='.env' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='.idea/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache/' \
  --exclude='assistant.db' \
  --exclude='chroma_data/' \
  "$LOCAL_DIR/" \
  "$REMOTE_HOST:$REMOTE_DIR/"

echo "[3/4] 验证并部署新版本"
ssh "$REMOTE_HOST" "
  set -e
  cd '$REMOTE_DIR'
  docker compose config >/dev/null
  docker compose up -d --build --wait --wait-timeout 120
"

echo "[4/4] 查看运行状态"
ssh "$REMOTE_HOST" "cd '$REMOTE_DIR' && docker compose ps"

echo "部署完成"
