#!/usr/bin/env bash
# Restart the AgentMart A2A server (systemd user service) and wait for it to
# accept connections on 127.0.0.1:9901.
set -euo pipefail

SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
APPDIR="$(dirname "$(dirname "${SCRIPT}")")"
REPO_ROOT="$(dirname "$(dirname "${APPDIR}")")"
SERVICE_NAME="agentmart-a2a"
PORT="9901"
UNIT_SRC="${APPDIR}/systemd/${SERVICE_NAME}.service"
UNIT_DST="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/${SERVICE_NAME}.service"

# Install the unit from the repo on first use so a fresh checkout is self-healing.
# The ExecStart path is injected at install time: the repo copy uses %h as a
# placeholder so it stays portable, and this substitutes the real checkout path.
if [ ! -f "${UNIT_DST}" ] || ! grep -qF "WorkingDirectory=${APPDIR}" "${UNIT_DST}" 2>/dev/null; then
    echo "installing ${SERVICE_NAME} unit from repo (appdir=${APPDIR})..."
    mkdir -p "$(dirname "${UNIT_DST}")"
    sed "s|%h/Building-Autonomous-AI-Agent|${REPO_ROOT}|g; s|%h|${HOME}|g" "${UNIT_SRC}" > "${UNIT_DST}"
    systemctl --user daemon-reload
    systemctl --user enable "${SERVICE_NAME}"
fi

echo "restarting ${SERVICE_NAME}..."
systemctl --user restart "${SERVICE_NAME}"

echo -n "waiting for port ${PORT}..."
for _ in $(seq 1 20); do
    if curl -s -m 2 "http://127.0.0.1:${PORT}/.well-known/agent-card.json" >/dev/null 2>&1; then
        echo " ok"
        systemctl --user --no-pager status "${SERVICE_NAME}" | head -n 5
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo " failed (port ${PORT} not accepting)"
systemctl --user --no-pager status "${SERVICE_NAME}"
exit 1