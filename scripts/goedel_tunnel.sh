#!/bin/bash
# Bring up Goedel-Prover-V2-32B (bf16, vLLM) on the A6000 box and tunnel it here.
#
#   scripts/goedel_tunnel.sh up      start remote server (if down) + local tunnel
#   scripts/goedel_tunnel.sh status  where things stand
#   scripts/goedel_tunnel.sh down    close the tunnel (leaves the server running)
#   scripts/goedel_tunnel.sh stop    stop the REMOTE server too
#
# Why vLLM and not the server's ollama: vLLM honours `n` per request, so one
# call returns N independent proofs sharing the prompt's KV cache. Goedel-V2's
# 88.1% MiniF2F is pass@32; sampling once throws that away.
#
# NEVER pkill -f a pattern that also matches this script's own command line --
# doing exactly that once killed someone else's training run on a shared server. PIDs are recorded
# to files and killed by PID.
set -u

GOEDEL_HOST="${GOEDEL_HOST:-$(sed -n 's/^GOEDEL_HOST=//p' "$(dirname "$0")/../.env" 2>/dev/null | tail -1 || true)}"
HOST="${GOEDEL_HOST:?set GOEDEL_HOST (in the environment or .env) to the GPU server ssh host}"
PORT="${GOEDEL_PORT:-8001}"
# Which serve script this tunnel drives. Default keeps the Goedel behaviour;
# SERVE_SCRIPT=serve_formalizer.sh GOEDEL_PORT=8002 brings up the tuned
# formalizer on GPU 0, which the prover on GPUs 1,2 does not use.
SERVE_SCRIPT="${SERVE_SCRIPT:-serve_goedel.sh}"
PIDFILE="/tmp/goedel_tunnel_${PORT}.pid"

# `timeout` wraps the WHOLE ssh call. ConnectTimeout bounds only the TCP
# handshake, so a session that connects and then hangs blocks forever -- that
# wedged the formalizer A/B for 20 minutes while the server was already up.
remote_up() { timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=10 \
  -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "$HOST" \
  "curl -s -m 5 http://127.0.0.1:${PORT}/v1/models >/dev/null 2>&1" ; }
local_up()  { curl -s -m 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 ; }

case "${1:-up}" in
up)
  if remote_up; then
    echo "remote: already serving on ${PORT}"
  else
    # DEPLOY the serve script from the repo rather than trusting whatever
    # happens to be in the remote home directory. The repo is the source of
    # truth: the working version of this script (with the FlashInfer/nvcc
    # workaround) existed only on the server for a while, which meant the
    # setup could not be reproduced from a clean checkout.
    echo "remote: deploying scripts/${SERVE_SCRIPT}"
    scp -q -o BatchMode=yes "$(dirname "$0")/${SERVE_SCRIPT}" \
        "$HOST:~/${SERVE_SCRIPT}" && ssh -o BatchMode=yes "$HOST" "chmod +x ~/${SERVE_SCRIPT}"
    echo "remote: starting vLLM (loads 62 GB, ~60-90s)..."
    ssh -o BatchMode=yes "$HOST" \
      "cd ~ && setsid nohup bash ~/${SERVE_SCRIPT} > ~/serve_${PORT}.log 2>&1 < /dev/null & echo \$! > ~/serve_${PORT}.pid"
    # Progress on EVERY check. The previous version printed nothing between
    # "starting" and "ready", so a normal 6-minute 62 GB load was
    # indistinguishable from a hang -- and each remote_up opens a fresh SSH
    # connection, so the real wall-clock is well past the 400s the loop counter
    # implies.
    for i in $(seq 1 40); do
      sleep 10
      if remote_up; then echo "remote: ready after ~$((i*10))s of polling"; break; fi
      echo "   ...loading (check $i/40; 62 GB across 2 GPUs takes ~6 min)"
      if [ "$i" = 40 ]; then
        echo "remote: FAILED to come up -- root cause:"
        ssh -o BatchMode=yes "$HOST" "grep -iE 'ValueError|RuntimeError' ~/serve_${PORT}.log | tail -3"
        exit 1
      fi
    done
  fi

  if local_up; then
    echo "tunnel: already open"
  else
    ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 -L "${PORT}:127.0.0.1:${PORT}" "$HOST"
    sleep 2
    pgrep -f "${PORT}:127.0.0.1:${PORT}" | head -1 > "$PIDFILE"
    echo "tunnel: open (pid $(cat "$PIDFILE" 2>/dev/null))"
  fi
  local_up && echo "OK: $(curl -s http://127.0.0.1:${PORT}/v1/models | head -c 90)..."
  ;;

status)
  remote_up && echo "remote: UP" || echo "remote: down"
  local_up  && echo "tunnel: UP" || echo "tunnel: down"
  ssh -o BatchMode=yes "$HOST" \
    'nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader' 2>/dev/null
  ;;

down)
  if [ -f "$PIDFILE" ]; then kill "$(cat "$PIDFILE")" 2>/dev/null && echo "tunnel closed"; rm -f "$PIDFILE"
  else echo "no recorded tunnel pid"; fi
  ;;

stop)
  ssh -o BatchMode=yes "$HOST" \
    'P=$(cat ~/serve_'"${PORT}"'.pid 2>/dev/null); [ -n "$P" ] && kill "$P" 2>/dev/null && echo "remote stopped (pid $P)" || echo "no recorded remote pid"'
  "$0" down
  ;;

*) echo "usage: $0 {up|status|down|stop}"; exit 2 ;;
esac
