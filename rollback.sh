#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE_HOST="myserver"
REMOTE_DIR="/home/admin/projects/personal-assistant-agent"
BACKUP_ARCHIVE="/home/admin/deploy-backups/personal-assistant/source-rollback.tar.gz"

CURRENT_IMAGE="personal-assistant-agent:dev"
ROLLBACK_IMAGE="personal-assistant-agent:rollback"

echo "[1/3] 检查回滚版本"
ssh "$REMOTE_HOST" "
  test -f '$BACKUP_ARCHIVE'
  docker image inspect '$ROLLBACK_IMAGE' >/dev/null
"

echo "[2/3] 恢复旧代码、配置和镜像"
ssh "$REMOTE_HOST" "
  set -e
  temp_dir=\$(mktemp -d)
  trap 'rm -rf \"\$temp_dir\"' EXIT

  tar -xzf '$BACKUP_ARCHIVE' -C \"\$temp_dir\"
  rsync -a --delete --exclude='.env' \"\$temp_dir/\" '$REMOTE_DIR/'
  docker image tag '$ROLLBACK_IMAGE' '$CURRENT_IMAGE'

  cd '$REMOTE_DIR'
  docker compose config >/dev/null
  docker compose up -d --no-build --force-recreate --wait --wait-timeout 120
"

echo "[3/3] 查看恢复状态"
ssh "$REMOTE_HOST" "cd '$REMOTE_DIR' && docker compose ps"

echo "回滚完成"
