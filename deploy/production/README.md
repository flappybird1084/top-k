# Production gateway

The container is the durable control plane for `top-k.dev`. It stores GitHub
sessions, per-user marimo and W&B credentials, run records, fetched artifacts,
and relay usage below `TOPK_DATA_ROOT`. Put that directory on an encrypted
volume. Training and kernel measurement run on the marimo GPU connected by
that signed-in user. The user's W&B key supplies both W&B Inference for the
planner/sub-agents and the observability destination; operator LLM relays stay
disabled.

The gateway is not published on an EC2 port. A Cloudflare Tunnel sidecar is the
only route into the private Compose network, and the Python gateway still
requires the Worker edge secret on every request.

Deployment:

1. Copy `.env.example` to `.env`, fill the secrets, and set mode `0600`. Keep
   the real file on the encrypted data volume and symlink `.env` to it.
2. Configure the Cloudflare tunnel hostname to route to `http://gateway:8768`.
3. Run `docker compose up -d --build`.
4. Set the `top-k-gateway` Worker `GPU_ORIGIN` secret to the tunnel hostname.

Back up both directories below `TOPK_DATA_ROOT`. `state` contains sign-ins and
per-user integration credentials; `jobs` contains run state, logs, and
artifacts.
