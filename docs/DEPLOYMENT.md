# Deployment

AutoML-Lens is deployed as a single Azure virtual machine that serves both the
React frontend and the FastAPI backend from one public HTTPS origin.

**Live URL:** `https://automl-lens.172-198-77-217.sslip.io`

---

## Architecture

```
browser
  │  HTTPS (TLS 1.2/1.3, Let's Encrypt)
  ▼
nginx  (172.198.77.217:80,443)
  ├── /                 → /var/www/automl-lens      built Vite SPA
  ├── /assets/*         → static, immutable 1-year cache
  └── /api/*            → 127.0.0.1:8000            FastAPI (uvicorn)
                              ├── SQLite     backend/storage/automl.db
                              └── artifacts  backend/storage/{reports,predictions,models}
```

Because the SPA and the API share one origin, the browser never makes a
cross-origin request, CORS does not apply, and `VITE_API_BASE_URL` stays empty
(the frontend calls relative `/api/...`).

## Azure resources

| Item | Value |
|---|---|
| Subscription | Azure for Students |
| Resource group | `automl-lens-rg` |
| Region | `indiasouthcentral` (Hyderabad) |
| VM | `automl-lens`, Ubuntu 22.04, `Standard_B2s` (2 vCPU, 4 GiB RAM) |
| OS disk | 30 GiB `Standard_LRS` managed disk (persistent) |
| Public IP | `172.198.77.217`, **Static** (Standard SKU) |
| NSG | 22 (SSH), 80 (HTTP), 443 (HTTPS) |
| Service | `automl-lens.service` (systemd) |
| Website root | `/var/www/automl-lens` |
| App directory | `/opt/automl-lens` (git clone of `main`) |
| Secrets | `/etc/automl-lens.env` (root-owned, never in git) |

> **Region restriction.** Azure for Students enforces the subscription-level
> `sys.regionrestriction` policy (`deny`) that permits only
> `koreacentral`, `uaenorth`, `indiasouthcentral`, `eastasia`, and
> `malaysiawest`. Deploying anywhere else fails with HTTP 400 while the az CLI
> reports the confusing *"content for this response was already consumed"*.

### Hostname

The hostname uses **sslip.io**, a wildcard DNS service that resolves
`<label>.<dashed-ip>.sslip.io` to an IP address. It needs no domain purchase,
no DNS account, and no payment method. Because the VM's IP is **static**, the
hostname stays valid across restarts and deallocations.

A real domain can be substituted: point an A record at `172.198.77.217`, then
re-run the TLS step with the new name.

## One worker is mandatory

`deploy/systemd/automl-lens.service` runs uvicorn with **one worker on purpose**:

```ini
ExecStart=.../uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Training is dispatched with `await asyncio.to_thread(...)` and progress is
tracked in an **in-process** registry. A second worker would hold a second,
independent copy of that state, so `/progress` would report "no training in
progress" while the other worker trained. Scale vertically (a bigger SKU)
rather than adding workers.

## Provisioning

```bash
git clone https://github.com/SruthiRagyari/AUTOML-LENS.git
cd AUTOML-LENS
sudo ./deploy/install_backend.sh <your-domain>
```

The script installs Python 3.11, nginx, certbot, creates the `automl-lens`
service user and virtualenv, clones the app, installs
`backend/requirements.txt`, writes `/etc/automl-lens.env`, registers the
systemd unit, and then:

1. bootstraps a **plain-HTTP** nginx server block (the committed TLS config
   references certificate files that do not exist yet, so `nginx -t` fails),
2. runs `certbot certonly --webroot` to issue the certificate,
3. installs the real TLS config only once the certificate exists.

If issuance fails the site still comes up over HTTP, so provisioning never
## TLS renewal

`certbot.timer` is enabled and renews automatically:

```bash
sudo systemctl list-timers certbot.timer
sudo certbot renew --dry-run
```

The certificate was issued with `--register-unsafely-without-email`, so no
contact address is on file and **there are no expiry emails** — watch the timer
instead. To add one later:

```bash
sudo certbot register --email you@example.com
```

Let's Encrypt rejects reserved placeholder domains such as `example.com`
(`invalidContact`), so never pass those.

## Frontend deployment

The SPA is built locally and copied to the server:

```bash
cd frontend && npm run build
tar -czf /tmp/dist.tar.gz -C dist .
scp /tmp/dist.tar.gz azureuser@172.198.77.217:/tmp/
# on the VM:
sudo rm -rf /var/www/automl-lens/*
sudo tar -xzf /tmp/dist.tar.gz -C /var/www/automl-lens
sudo chown -R www-data:www-data /var/www/automl-lens
```

To deploy to Vercel instead, set `VITE_API_BASE_URL` to this backend's origin,
add the resulting Vercel origin to `CORS_ORIGINS`, and `frontend/vercel.json`
already carries the SPA rewrite.

## Configuration

`/etc/automl-lens.env` is created from `deploy/env.production.example`:

| Variable | Purpose |
|---|---|
| `APP_ENV=production` | production behaviour |
| `DATABASE_PATH`, `STORAGE_DIR` | persistent locations on the managed disk |
| `MAX_DATASET_SIZE_MB=1024` | 1 GB upload ceiling (nginx allows 2048m) |
| `CORS_ORIGINS` | only needed for a split frontend; same-origin ignores it |
| `LLM_PROVIDER` | `gemini` / `openai`, or unset for the deterministic fallback |
| `GEMINI_API_KEY`, `OPENAI_API_KEY` | set on the server only, never in git |

The AI Assistant works **without any API key** by falling back to the
deterministic local engine, which labels its own answers honestly. Adding a
provider key is optional and is entered only on the server.

## Operations

```bash
sudo systemctl status automl-lens
sudo journalctl -u automl-lens -f
sudo systemctl restart automl-lens
curl https://<domain>/api/health
```

Update code:

```bash
sudo git config --global --add safe.directory /opt/automl-lens   # once
sudo bash -c "cd /opt/automl-lens && git pull --ff-only origin main"
sudo systemctl restart automl-lens
```

### Backup

Everything durable lives under `/opt/automl-lens/backend/storage/`. Stop the
service first for a consistent copy of the SQLite database:

```bash
sudo systemctl stop automl-lens
sudo tar -czf /tmp/backup.tgz -C /opt/automl-lens backend/storage
sudo systemctl start automl-lens
```

## Cost control

The site runs 24/7 unless stopped, and an Azure VM accrues compute cost for
every minute it is **allocated** (running *or* stopped). **Deallocate** it to
stop compute billing:

```bash
az vm deallocate -g automl-lens-rg -n automl-lens   # stops compute billing
az vm start     -g automl-lens-rg -n automl-lens   # brings it back
```

Auto-shutdown at 23:30 IST is configured with:

```bash
az vm auto-shutdown -g automl-lens-rg -n automl-lens \
  --time 2330 --timezone "India Standard Time" --status Enabled
```

**The site is offline from 23:30 until it is next started.** Disabling the rule
costs more but keeps the site always reachable.

Cost while running is **estimated at roughly $38/month** (VM + static IPv4 +
30 GiB managed disk). Treat that as an estimate — it was not measured from an
actual invoice during this deployment.

## Security notes

- The VM is reachable on 22/80/443. Restrict 22 to your own IP if you prefer.
- `/etc/automl-lens.env` is the only place secrets live; it is never committed.
- The application has **no authentication** — anyone who reaches the URL can
  upload data and start training. It is meant as a public demo.

## Verified at deployment time

Real public end-to-end run against the live site: dataset upload (300 rows x 5
cols) → profiling → target selection → AutoML training (4 models, 20 Optuna
trials each, 5 folds, 64 s) → model selection (Gradient Boosting, f1_weighted)
→ 3 ensemble candidates → SHAP explainability → single and batch predictions →
report, model (306,717 B) and metadata downloads → AI Assistant in both
general and project context. All 10 SPA routes returned HTTP 200 with zero
console errors and zero `localhost` requests.
After editing:

```bash
sudo systemctl restart automl-lens
```
aborts mid-way.