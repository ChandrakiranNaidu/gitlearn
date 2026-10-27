#!/bin/bash
# Startup script for an Ubuntu 24.04 server (Vultr "Atlanta" or Linode "Atlanta, GA").
# Paste it into the provider's "Startup Script" / "User Data" box when creating the server.
# Edit the values below first.
set -euo pipefail

REPO_URL="https://github.com/chandrakirannaidu/gitlearn.git"
BRANCH="ccr-429e51c9-yd5wig"    # branch that contains the app
APP_SUBDIR="qr-survey"          # folder inside the repo ("" if the app is at the repo root)
GITHUB_TOKEN=""              # needed only if the repo is private (fine-grained token, read-only "Contents")
ADMIN_PASSWORD="change-me"   # password for /admin (username can be anything)

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y curl git debian-keyring debian-archive-keyring apt-transport-https gnupg

# Node.js 22
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs

# Caddy (automatic HTTPS)
curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt > /etc/apt/sources.list.d/caddy-stable.list
apt-get update && apt-get install -y caddy

# App
CLONE_URL="$REPO_URL"
[ -n "$GITHUB_TOKEN" ] && CLONE_URL="${REPO_URL/https:\/\//https://x-access-token:${GITHUB_TOKEN}@}"
git clone --depth 1 --branch "$BRANCH" "$CLONE_URL" /opt/src
ln -s "/opt/src/${APP_SUBDIR}" /opt/qr-survey
cd /opt/qr-survey && npm ci --omit=dev

# Free hostname that resolves to this server's IP, so HTTPS works without buying a domain.
IP=$(curl -fsS https://api.ipify.org)
HOST="${IP//./-}.sslip.io"

cat > /etc/systemd/system/qr-survey.service <<EOF
[Unit]
Description=QR survey
After=network.target

[Service]
WorkingDirectory=/opt/qr-survey
Environment=PORT=3000
Environment=PUBLIC_URL=https://${HOST}
Environment=ADMIN_PASSWORD=${ADMIN_PASSWORD}
Environment=TOKEN_SECRET=$(openssl rand -hex 32)
ExecStart=/usr/bin/node server.js
Restart=always

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/caddy/Caddyfile <<EOF
${HOST} {
  reverse_proxy localhost:3000
}
EOF

systemctl daemon-reload
systemctl enable --now qr-survey
systemctl restart caddy
ufw allow 22,80,443/tcp >/dev/null 2>&1 || true

echo "Survey live at https://${HOST}  (admin: https://${HOST}/admin)" | tee /root/SURVEY_URL.txt
