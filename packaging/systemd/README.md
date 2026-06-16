# Systemd deployment for rtsp-warden

## Install

```bash
# 1. Install rtsp-warden system-wide (one-time)
sudo pip install rtsp-warden  # or use uv tool install

# 2. Run the install script
sudo packaging/systemd/install.sh

# 3. Configure
sudo cp packaging/systemd/warden.env.example /etc/rtsp-warden/warden.env
sudo $EDITOR /etc/rtsp-warden/warden.env

sudo cp /path/to/your/config.yaml /etc/rtsp-warden/config.yaml
sudo $EDITOR /etc/rtsp-warden/config.yaml

# 4. Enable + start
sudo systemctl enable --now rtsp-warden

# 5. Check status
sudo systemctl status rtsp-warden
sudo journalctl -u rtsp-warden -f
```

## Updating

```bash
sudo pip install --upgrade rtsp-warden
sudo systemctl restart rtsp-warden
```

## Uninstall

```bash
sudo packaging/systemd/uninstall.sh              # keep data
sudo packaging/systemd/uninstall.sh --purge-data # remove recordings
```

## File locations

| Path | Purpose |
|------|---------|
| `/usr/local/bin/rtsp-warden` | Executable (from `pip install`) |
| `/etc/rtsp-warden/config.yaml` | Main config |
| `/etc/rtsp-warden/warden.env` | Environment overrides |
| `/var/lib/rtsp-warden/recordings` | Video segments |
| `/var/lib/rtsp-warden/data` | SQLite DB (if using SQLite) |
| `/var/log/rtsp-warden` | Logs (also in journal) |
| `/etc/systemd/system/rtsp-warden.service` | Service unit |