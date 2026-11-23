#!/usr/bin/env bash
# Put Fairview online on a fresh Ubuntu 24.04 server.
#
#   sudo bash setup.sh fairview.tech
#
# Before running it, point the domain's A record (and www) at this server's
# IP address, so Caddy can obtain the HTTPS certificate. Run it again later to
# update the site to the latest code.
set -euo pipefail

DOMAIN="${1:?usage: sudo bash setup.sh your-domain}"
REPO_URL="${REPO_URL:-https://github.com/fukubie/fairview.git}"
DIR=/opt/fairview

apt-get update
apt-get install -y docker.io docker-compose-v2 git
systemctl enable --now docker

if [ -d "$DIR/.git" ]; then
    git -C "$DIR" pull --ff-only
else
    git clone --depth 1 "$REPO_URL" "$DIR"
fi

cd "$DIR"
DOMAIN="$DOMAIN" docker compose -f deploy/server/docker-compose.yml up -d --build

echo
echo "Fairview is starting at https://$DOMAIN"
echo "Logs: cd $DIR && DOMAIN=$DOMAIN docker compose -f deploy/server/docker-compose.yml logs -f"
