#!/bin/bash
# Serves a model with vLLM on a remote machine and forwards its port to localhost.
#
# Usage: ./bash/scripts/serve_model.sh <model> <user@remote-host> [<port>]
#   e.g. ./bash/scripts/serve_model.sh google/gemma-3-4b-it aswathy@gpu-box 8000
#
# Set REMOTE_SETUP to a command that makes `vllm` available on the remote machine,
# e.g. REMOTE_SETUP="conda activate vllm-env".

model=$1
remote=$2
port=${3:-8000}

if [ -z "$model" ] || [ -z "$remote" ]; then
    echo "Usage: $0 <model> <user@remote-host> [<port>]"
    exit 1
fi

log="vllm_${port}.log"
pidfile="vllm_${port}.pid"

set -e

# Close the tunnel if the script exits before the server is ready (error or Ctrl-C)
tunnel="ssh .*-L $port:localhost:$port $remote"
ready=0
cleanup() {
    if [ "$ready" -eq 0 ]; then
        pkill -f "$tunnel" 2> /dev/null || true
    fi
}
trap cleanup EXIT

echo "Starting vLLM for $model on $remote:$port"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$remote" "bash -lc '
    ${REMOTE_SETUP:+$REMOTE_SETUP &&}
    nohup vllm serve $model --host 127.0.0.1 --port $port > $log 2>&1 &
    echo \$! > $pidfile
'"

echo "Forwarding localhost:$port to $remote:$port"
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ExitOnForwardFailure=yes -f -N -L "$port:localhost:$port" "$remote"

echo "Waiting for the server (model loading can take a few minutes)"
until curl -sf "localhost:$port/v1/models" > /dev/null; do
    if ! ssh -o BatchMode=yes "$remote" "kill -0 \$(cat $pidfile)" 2> /dev/null; then
        echo "vLLM exited on $remote. Last log lines:"
        ssh "$remote" "tail -n 20 $log"
        exit 1
    fi
    sleep 5
done
ready=1
echo "Ready: http://localhost:$port/v1"

echo "Logs: ssh $remote tail -f $log"
echo "Stop: ssh $remote 'kill \$(cat $pidfile)'; pkill -f '$tunnel'"
