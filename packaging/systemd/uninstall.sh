#!/usr/bin/env bash
set -euo pipefail

PURGE_DATA=false
if [[ "${1:-}" == "--purge-data" ]]; then
    PURGE_DATA=true
fi

SERVICE_USER="rtsp-warden"
SERVICE_GROUP="rtsp-warden"

echo "==> Uninstalling rtsp-warden systemd service..."

# 1. Stop and disable the service
if systemctl is-active rtsp-warden &>/dev/null; then
    echo "    Stopping rtsp-warden service..."
    systemctl stop rtsp-warden
else
    echo "    Service is not running, skipping stop."
fi

if systemctl is-enabled rtsp-warden &>/dev/null; then
    echo "    Disabling rtsp-warden service..."
    systemctl disable rtsp-warden
else
    echo "    Service is not enabled, skipping disable."
fi

# 2. Remove service file
if [[ -f /etc/systemd/system/rtsp-warden.service ]]; then
    echo "    Removing /etc/systemd/system/rtsp-warden.service..."
    rm -f /etc/systemd/system/rtsp-warden.service
else
    echo "    Service file not found, skipping removal."
fi

# 3. Reload systemd
echo "==> Reloading systemd daemon..."
systemctl daemon-reload

# 4. Remove directories
echo "==> Removing directories..."

if [[ -d /etc/rtsp-warden ]]; then
    rm -rf /etc/rtsp-warden
    echo "    Removed /etc/rtsp-warden"
else
    echo "    /etc/rtsp-warden not found, skipping."
fi

if [[ -d /var/log/rtsp-warden ]]; then
    rm -rf /var/log/rtsp-warden
    echo "    Removed /var/log/rtsp-warden"
else
    echo "    /var/log/rtsp-warden not found, skipping."
fi

if [[ -d /var/lib/rtsp-warden ]]; then
    if ${PURGE_DATA}; then
        rm -rf /var/lib/rtsp-warden
        echo "    Removed /var/lib/rtsp-warden (including recordings)"
    else
        echo "    Keeping /var/lib/rtsp-warden (use --purge-data to remove recordings)"
    fi
else
    echo "    /var/lib/rtsp-warden not found, skipping."
fi

# 5. Remove user and group
if id "${SERVICE_USER}" &>/dev/null; then
    echo "==> Removing system user '${SERVICE_USER}'..."
    userdel "${SERVICE_USER}" 2>/dev/null || true
else
    echo "    User '${SERVICE_USER}' does not exist, skipping."
fi

if getent group "${SERVICE_GROUP}" &>/dev/null; then
    echo "==> Removing system group '${SERVICE_GROUP}'..."
    groupdel "${SERVICE_GROUP}" 2>/dev/null || true
else
    echo "    Group '${SERVICE_GROUP}' does not exist, skipping."
fi

echo ""
echo "==> Uninstall complete."
if ! ${PURGE_DATA}; then
    echo "    Note: /var/lib/rtsp-warden was preserved. Re-run with --purge-data to remove."
fi