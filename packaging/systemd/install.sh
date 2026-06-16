#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_USER="rtsp-warden"
SERVICE_GROUP="rtsp-warden"

echo "==> Installing rtsp-warden systemd service..."

# 1. Create system user (idempotent)
if id "${SERVICE_USER}" &>/dev/null; then
    echo "    User '${SERVICE_USER}' already exists, skipping."
else
    useradd --system --no-create-home --shell /usr/sbin/nologin "${SERVICE_USER}"
    echo "    Created system user '${SERVICE_USER}'."
fi

# 2. Create directories
echo "==> Creating directories..."

install -d -m 0755 -o root -g "${SERVICE_GROUP}" /etc/rtsp-warden
echo "    /etc/rtsp-warden  (mode 755, root:${SERVICE_GROUP})"

install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" /var/lib/rtsp-warden
install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" /var/lib/rtsp-warden/recordings
echo "    /var/lib/rtsp-warden  (mode 750, ${SERVICE_USER}:${SERVICE_GROUP})"
echo "    /var/lib/rtsp-warden/recordings  (mode 750, ${SERVICE_USER}:${SERVICE_GROUP})"

install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" /var/log/rtsp-warden
echo "    /var/log/rtsp-warden  (mode 750, ${SERVICE_USER}:${SERVICE_GROUP})"

# 3. Copy service file
echo "==> Installing systemd unit..."
install -m 0644 "${SCRIPT_DIR}/rtsp-warden.service" /etc/systemd/system/rtsp-warden.service
echo "    /etc/systemd/system/rtsp-warden.service"

# 4. Reload systemd
echo "==> Reloading systemd daemon..."
systemctl daemon-reload

echo ""
echo "==> Installation complete."
echo ""
echo "    Next steps:"
echo "      1. Copy your config: sudo cp /path/to/config.yaml /etc/rtsp-warden/config.yaml"
echo "      2. (Optional) Copy env file: sudo cp ${SCRIPT_DIR}/warden.env.example /etc/rtsp-warden/warden.env"
echo "      3. Enable and start:   sudo systemctl enable --now rtsp-warden"
echo "      4. Check status:       sudo systemctl status rtsp-warden"
echo "      5. View logs:          sudo journalctl -u rtsp-warden -f"