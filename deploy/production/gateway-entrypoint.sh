#!/bin/sh
# The gateway runs unprivileged as `topk`, but its two data directories are
# bind-mounted from the host and Docker brings a bind-mount source up owned by
# root. That hides the image's pre-owned directories and leaves `topk` unable
# to create its auth DB, integration store, or job files — so the process
# would crash on the first request. Start as root only long enough to hand the
# mounts back to `topk`, then drop privileges and exec the gateway.
set -e

chown -R topk:topk /var/lib/top-k /app/jobs

exec gosu topk "$@"
