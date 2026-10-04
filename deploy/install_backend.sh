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

if [[ -z "$DOMAIN" ]]; then
  echo "usage: $0 <domain> [repo-url]" >&2
  exit 1
fi

echo "==> Domain: $DOMAIN"

# ── 1. System packages ────────────────────────────────────────────────────────
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  git nginx software-properties-common ca-certificates curl

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

# ── 8. nginx reverse proxy ────────────────────────────────────────────────────
echo "==> Configuring nginx for $DOMAIN"
sed -e "s/DOMAIN/$DOMAIN/g" "$APP_DIR/deploy/nginx/automl-lens.conf" > /etc/nginx/sites-available/automl-lens
ln -sf /etc/nginx/sites-available/automl-lens /etc/nginx/sites-enabled/automl-lens
rm -f /etc/nginx/sites-enabled/default
nginx -t

echo "==> Starting nginx (HTTP only; TLS is added by certbot in the next step)"
systemctl enable --now nginx
systemctl reload nginx

cat <<EOF

Next steps (run as root):

  1. Point your DNS A record for $DOMAIN at this server's public IP.
  2. Issue the TLS certificate:
       apt-get install -y certbot python3-certbot-nginx
       certbot --nginx -d $DOMAIN --agree-tos -m you@example.com --redirect
  3. Edit secrets (NOT in git):
       $ENV_FILE
     Set at minimum:
       LLM_PROVIDER=gemini   (or 'fallback' for the deterministic offline provider)
       GEMINI_API_KEY=<your key>
       CORS_ORIGINS=<the frontend origin, e.g. https://your-app.vercel.app>
  4. Apply and check:
       systemctl restart automl-lens
       curl https://$DOMAIN/api/health

EOF