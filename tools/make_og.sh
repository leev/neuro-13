#!/bin/sh
# Render tools/og.html to src/assets/og.png (1200x630 link preview). It trains the home page's network first, so renders differ a little.
cd "$(dirname "$0")/.." || exit 1
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CHROME" --headless=new --disable-gpu --no-first-run --hide-scrollbars \
  --user-data-dir="${TMPDIR:-/tmp}/neuro13-og" --force-prefers-reduced-motion \
  --window-size=1200,630 --virtual-time-budget=8000 \
  --screenshot="$PWD/src/assets/og.png" "file://$PWD/tools/og.html" 2>/dev/null
python3 -c "
import struct, sys
with open('src/assets/og.png', 'rb') as f:
    w, h = struct.unpack('>II', f.read(24)[16:24])
print(f'src/assets/og.png {w}x{h}')
sys.exit(0 if (w, h) == (1200, 630) else 1)"
