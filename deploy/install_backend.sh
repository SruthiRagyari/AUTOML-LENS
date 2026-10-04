#!/usr/bin/env bash
#
# AutoML-Lens backend provisioning for a single Ubuntu 22.04 VM (OCI Ampere A1 or
# any x86/arm64 Linux box). Installs the runtime, the app, the systemd unit and the
# nginx TLS reverse proxy.
#
# Usage:
#   sudo ./deploy/install_backend.sh <domain> [repo-url]
#
# Example:
#   sudo ./deploy/install_backend.sh api.automl-lens.example \
#        https://github.com/SruthiRagyari/AUTOML-LENS.git
#
# The script is idempotent: re-running it fast-forwards the app and restarts it.
set -euo pipefail

DOMAIN="${1:-}"
REPO_URL="${2:-https://github.com/SruthiRagyari/AUTOML-LENS.git}"
APP_DIR=/opt/automl-lens
SERVICE_USER=automl-lens
ENV_FILE=/etc/automl-lens.env
# Used by certbot for the ACME account and expiry notices. Override if wanted:
#   LETSENCRYPT_EMAIL=you@example.com sudo ./deploy/install_backend.sh <domain>
LETSENCRYPT_EMAIL="${LETSENCRYPT_EMAIL:-admin@example.com}"

if [[ -z "$DOMAIN" ]]; then
  echo "usage: $0 <domain> [repo-url]" >&2
  exit 1
fi

echo "==> Domain: $DOMAIN"

# ── 1. System packages ────────────────────────────────────────────────────────
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  git nginx software-properties-common ca-certificates curl certbot

# Python 3.11 to match the wheels verified for this project (Ubuntu 22.04 ships 3.10).
if ! command -v python3.11 >/dev/null 2>&1; then
  echo "==> Adding deadsnakes PPA for Python 3.11"
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update -y
fi
apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev build-essential

# ── 2. Service account ────────────────────────────────────────────────────────
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
  echo "==> Creating service user $SERVICE_USER"
  useradd --system --create-home --home-dir /home/"$SERVICE_USER" --shell /usr/sbin/nologin "$SERVICE_USER"
fi

# ── 3. Application code ───────────────────────────────────────────────────────
if [[ -d "$APP_DIR/.git" ]]; then
  echo "==> Updating application"
  git -C "$APP_DIR" fetch --all --tags
  git -C "$APP_DIR" checkout main
  git -C "$APP_DIR" pull --ff-only origin main
else
  echo "==> Cloning application"
  mkdir -p "$APP_DIR"
  git clone "$REPO_URL" "$APP_DIR"
fi
chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR"

# ── 4. Python environment ─────────────────────────────────────────────────────
echo "==> Creating virtualenv and installing dependencies (this takes a few minutes)"
python3.11 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip wheel
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/backend/requirements.txt"
chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR/venv"

# ── 5. Persistent data directories (SQLite + datasets/models/reports) ─────────
echo "==> Ensuring persistent storage directories"
mkdir -p "$APP_DIR/backend/storage"/{datasets,models,reports,predictions}
chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR/backend"

# ── 6. Environment file (secrets live here, never in git) ─────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
  echo "==> Creating $ENV_FILE from template"
  cp "$APP_DIR/deploy/env.production.example" "$ENV_FILE"
  chmod 640 "$ENV_FILE"
  chown root:"$SERVICE_USER" "$ENV_FILE"
fi

# ── 7. systemd service ────────────────────────────────────────────────────────
echo "==> Installing systemd unit"
install -m 644 "$APP_DIR/deploy/systemd/automl-lens.service" /etc/systemd/system/automl-lens.service
systemctl daemon-reload
systemctl enable automl-lens
systemctl restart automl-lens
sleep 5
systemctl --no-pager --full status automl-lens || true

# ── 8. nginx + TLS ───────────────────────────────────────────────────────────
# The committed site config references certificates that do not exist yet, so
# nginx must be bootstrapped on plain HTTP first (for the ACME challenge),
# then the certificate is issued, then the real TLS config is installed.
echo "==> Bootstrapping nginx on HTTP for ACME validation"
cat > /etc/nginx/sites-available/automl-lens <<NGINX
server {
    listen 80;
    server_name ${DOMAIN};

    location /.well-known/acme-challenge/ { root /var/www/html; }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host              \$host;
        proxy_set_header X-Real-IP         \$remote_addr;
        proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_request_buffering off;
        proxy_buffering off;
        proxy_connect_timeout 60s;
        proxy_send_timeout    3600s;
        proxy_read_timeout    3600s;
    }
}
NGINX

mkdir -p /var/www/html
ln -sf /etc/nginx/sites-available/automl-lens /etc/nginx/sites-enabled/automl-lens
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl enable --now nginx
systemctl reload nginx

echo "==> Requesting TLS certificate for $DOMAIN"
certbot certonly --webroot -w /var/www/html -d "$DOMAIN" \
  --agree-tos -m "${LETSENCRYPT_EMAIL:-admin@example.com}" --non-interactive || \
  echo "WARNING: certbot failed - HTTP still works, HTTPS will not be enabled."

if [[ -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]]; then
  echo "==> Installing TLS site config"
  sed -e "s/DOMAIN/$DOMAIN/g" "$APP_DIR/deploy/nginx/automl-lens.conf" \
    > /etc/nginx/sites-available/automl-lens
  systemctl enable certbot.timer >/dev/null 2>&1 || true
  nginx -t
  systemctl reload nginx
  echo "==> HTTPS enabled"
else
  echo "==> No certificate yet; site is available over HTTP only."
fi

cat <<EOF

AutoML-Lens backend provisioned.

  Public URL : https://$DOMAIN  (or http://$DOMAIN if TLS failed)
  Health     : https://$DOMAIN/api/health
  App dir    : $APP_DIR
  Env file   : $ENV_FILE   (edit for CORS_ORIGINS / LLM_PROVIDER / GEMINI_API_KEY)
  Service    : systemctl status automl-lens

EOF