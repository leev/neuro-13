#!/usr/bin/env python3
"""
Sync neuro*.py source files into the textareas inside neuro-13.html.

Source of truth: the .py files on disk.
HTML is a generated artifact (the prose around the code is hand-written,
but the code inside <textarea id="py-code-N"> blocks comes from here).

Run this after editing any neuro*.py:

    python3 build.py
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HTML = HERE / 'neuro-13.html'

MAPPING = {
    'py-code-1': 'neuro1.py',
    'py-code-2': 'neuro2.py',
    'py-code-3': 'neuro3.py',
    'py-code-4': 'neuro4.py',
    'py-code-5': 'neuro5.py',
    'py-code-6': 'neuro6.py',
    'py-code-7': 'neuro7.py',
    'py-code-8': 'neuro8.py',
    'py-code-9': 'neuro9.py',
}


def html_escape_for_textarea(s: str) -> str:
    # Inside a <textarea>, the only sequence that breaks parsing is the
    # literal "</textarea". We also escape & so any future ampersand-named
    # entity in code (e.g. &amp;) survives a round trip.
    s = s.replace('&', '&amp;')
    s = s.replace('</textarea', '&lt;/textarea')
    return s


def main() -> int:
    if not HTML.exists():
        print(f'missing {HTML}', file=sys.stderr)
        return 1

    html = HTML.read_text()
    changed = 0

    for ta_id, fname in MAPPING.items():
        src = HERE / fname
        if not src.exists():
            print(f'missing source {src}', file=sys.stderr)
            return 1
        code = src.read_text().rstrip() + '\n'
        code_safe = html_escape_for_textarea(code)

        # Match <textarea ... id="ta_id" ...>...</textarea>
        pattern = re.compile(
            rf'(<textarea\b[^>]*\bid="{re.escape(ta_id)}"[^>]*>)(.*?)(</textarea>)',
            re.DOTALL,
        )
        m = pattern.search(html)
        if not m:
            print(f'no <textarea id="{ta_id}"> in {HTML.name}', file=sys.stderr)
            return 1

        old = m.group(2)
        if old.strip() == code_safe.strip():
            continue

        html = pattern.sub(lambda _m: _m.group(1) + code_safe + _m.group(3), html, count=1)
        changed += 1
        print(f'  updated #{ta_id} from {fname}')

    if changed:
        HTML.write_text(html)
        print(f'wrote {HTML.name} ({changed} change{"s" if changed != 1 else ""})')
    else:
        print('no changes — HTML already up to date')
    return 0


if __name__ == '__main__':
    sys.exit(main())
