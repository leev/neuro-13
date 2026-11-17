#!/bin/sh
# Load every built page in headless Chrome and report JavaScript errors.
#   tools/smoke.sh [site-dir] [port]
DIR="${1:-public}"
PORT="${2:-8791}"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
( cd "$DIR" && exec python3 -m http.server "$PORT" >/dev/null 2>&1 ) &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
sleep 1
fail=0
for page in "$DIR"/*.html; do
  name=$(basename "$page")
  errors=$("$CHROME" --headless=new --disable-gpu --no-first-run --user-data-dir="${TMPDIR:-/tmp}/neuro13-smoke" \
    --enable-logging=stderr --log-level=0 --virtual-time-budget=4000 --dump-dom \
    "http://localhost:$PORT/$name" 2>&1 >/dev/null | grep -E 'CONSOLE.*(Uncaught|Error|Failed to load|404)' | grep -v 'favicon\|og.png')
  if [ -n "$errors" ]; then
    fail=1
    echo "FAIL $name"
    echo "$errors" | sed 's/^.*CONSOLE([0-9]*)\] /    /' | cut -c1-240
  else
    echo "ok   $name"
  fi
done
exit $fail
