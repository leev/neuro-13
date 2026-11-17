#!/usr/bin/env python3
"""
Build the book into a static site.

    src/book.json            parts, chapters and extra pages, in reading order
    src/chapters/<slug>.html one chapter's prose and widget markup
    src/widgets/<slug>.js    that chapter's interactive code (optional)
    src/widgets/<slug>.css   that chapter's extra styles (optional)
    src/pages/<slug>.html    unnumbered pages (certificate, grown-ups)
    src/assets/              shared CSS, JS, fonts and images
    src/glossary.json        term -> definition (src/glossary.d/*.json are merged in)
    py/                      Python exercises and the datasets they read

    python3 build.py                 -> public/
    python3 build.py --out DIR       -> DIR/
    python3 build.py --draft         tolerate chapters that are not written yet

Inside a chapter or page, {{ch:slug}} becomes a link reading "chapter 7" (the
number is looked up, so chapters can be reordered freely), {{Ch:slug}}
capitalises it, and {{title:slug}} links using the chapter's title.
"""
import argparse
import hashlib
import html
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
PY = ROOT / 'py'
PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/'
CODEMIRROR = 'https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/'


def load_book():
    book = json.loads((SRC / 'book.json').read_text())
    chapters = []
    for number, part in enumerate(book['parts'], 1):
        part['number'] = number
        for ch in part['chapters']:
            ch['number'] = len(chapters) + 1
            ch['part'] = part
            chapters.append(ch)
    return book, chapters


def expand_refs(text, by_slug, where):
    def sub(m):
        kind, slug = m.group(1), m.group(2)
        if slug not in by_slug:
            sys.exit(f'{where}: unknown chapter "{slug}" in {m.group(0)}')
        ch = by_slug[slug]
        label = {
            'ch': f'chapter {ch["number"]}',
            'Ch': f'Chapter {ch["number"]}',
            'title': html.escape(ch['title']),
        }[kind]
        return f'<a class="chref" href="{slug}.html">{label}</a>'
    return re.sub(r'\{\{(ch|Ch|title):([\w-]+)\}\}', sub, text)


def fill_code(text, where):
    def sub(m):
        src = PY / m.group(2)
        if not src.exists():
            sys.exit(f'{where}: missing {src}')
        code = src.read_text().rstrip() + '\n'
        code = code.replace('&', '&amp;').replace('</textarea', '&lt;/textarea')
        return m.group(1) + code + m.group(4)
    return re.sub(r'(<textarea\b[^>]*\bdata-src="([^"]+)"[^>]*>)(.*?)(</textarea>)', sub, text, flags=re.S)


def plain_text(fragment):
    fragment = re.sub(r'<(script|style|textarea|svg)\b.*?</\1>', ' ', fragment, flags=re.S)
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', fragment))).strip()


def nav_tree(book, current=None):
    out = []
    for part in book['parts']:
        is_open = current is not None and current['part'] is part
        out.append(f'<details class="nav-part" style="--part:{part["color"]};--part-tint:{part["tint"]}"{" open" if is_open else ""}>')
        out.append(f'<summary><span class="nav-part-num">Part {part["number"]}</span><span class="nav-part-title">{html.escape(part["title"])}</span>'
                   f'<span class="nav-part-count" data-part-count="{part["id"]}"></span></summary><ol>')
        for ch in part['chapters']:
            cls = ' class="active" aria-current="page"' if ch is current else ''
            out.append(f'<li><a href="{ch["slug"]}.html" data-slug="{ch["slug"]}"{cls}><span class="nav-num">{ch["number"]}</span>'
                       f'<span class="nav-title">{html.escape(ch["title"])}</span></a></li>')
        out.append('</ol></details>')
    return '\n'.join(out)


def render(template, values):
    def sub(m):
        if m.group(1) not in values:
            sys.exit(f'template wants %%{m.group(1)}%% but the build did not supply it')
        return str(values[m.group(1)])
    return re.sub(r'%%(\w+)%%', sub, template)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'public'))
    ap.add_argument('--draft', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)

    book, chapters = load_book()
    by_slug = {ch['slug']: ch for ch in chapters}
    glossary = json.loads((SRC / 'glossary.json').read_text())
    for extra in sorted((SRC / 'glossary.d').glob('*.json')):
        glossary.update(json.loads(extra.read_text()))

    if out.exists():
        shutil.rmtree(out)
    skip = shutil.ignore_patterns('__pycache__', '.DS_Store')
    shutil.copytree(SRC / 'assets', out / 'assets', ignore=skip)
    shutil.copytree(SRC / 'widgets', out / 'assets' / 'widgets', ignore=skip)
    shutil.copytree(PY, out / 'py', ignore=skip)
    (out / '_headers').write_text('/py/*.py\n  Content-Type: text/plain; charset=utf-8\n')
    with zipfile.ZipFile(out / 'neuro13-code.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted((out / 'py').rglob('*')):
            if f.is_file():
                z.write(f, Path('neuro13') / f.relative_to(out / 'py'))
        for f in sorted((ROOT / 'tools').glob('make_*.py')):
            z.write(f, Path('neuro13') / 'tools' / f.name)

    digest = hashlib.sha1()
    for f in sorted((out / 'assets').rglob('*')):
        if f.is_file():
            digest.update(f.read_bytes())
    v = digest.hexdigest()[:8]

    def script(path):
        return f'<script src="{path}?v={v}"></script>'

    def css(path):
        return f'<link rel="stylesheet" href="{path}?v={v}">'

    common = {
        'book_title': html.escape(book['title']),
        'site_url': book['url'],
        'v': v,
        'total': len(chapters),
    }
    topbar = render((SRC / 'topbar.html').read_text(), common)
    head_tpl = (SRC / 'head.html').read_text()
    sidebar_tpl = (SRC / 'sidebar.html').read_text()
    base_scripts = [script('assets/book-data.js'), script('assets/shell.js')]
    if (SRC / 'assets' / 'search.js').exists():
        base_scripts.append(script('assets/search.js'))

    def frame(title, description, slug, current=None, extra_head=()):
        page_title = book['title'] + ' — ' + book['tagline'] if slug == 'index' else f'{title} · {book["title"]}'
        head = render(head_tpl, {
            **common,
            'page_title': html.escape(page_title),
            'description': html.escape(description),
            'page_url': book['url'] + ('/' if slug == 'index' else f'/{slug}'),
        })
        return {
            **common,
            'head': '\n'.join([head, *extra_head]),
            'topbar': topbar,
            'sidenav': render(sidebar_tpl, {'sidebar': nav_tree(book, current)}),
            'title': html.escape(title),
            'slug': slug,
        }

    page_tpl = (SRC / 'template.html').read_text()
    simple_tpl = (SRC / 'simple.html').read_text()
    missing, widgets, search = [], 0, []

    for i, ch in enumerate(chapters):
        slug, part = ch['slug'], ch['part']
        src = SRC / 'chapters' / f'{slug}.html'
        if src.exists():
            raw = src.read_text()
            body = fill_code(expand_refs(raw, by_slug, src.name), src.name)
        else:
            missing.append(slug)
            raw = body = '<p class="lede">This chapter is still being written.</p>'
        widgets += body.count('class="interactive"')
        search.append({
            'slug': slug, 'number': ch['number'], 'title': ch['title'], 'hook': ch['hook'],
            'headings': [plain_text(h) for h in re.findall(r'<h3[^>]*>(.*?)</h3>', raw, flags=re.S)],
            'text': plain_text(expand_refs(raw, by_slug, src.name)),
        })

        extra_head, scripts = [], list(base_scripts) + [script('assets/glossary.js')]
        if (SRC / 'widgets' / f'{slug}.css').exists():
            extra_head.append(css(f'assets/widgets/{slug}.css'))
        if (SRC / 'widgets' / f'{slug}.js').exists():
            scripts += [script('assets/shared.js'), script(f'assets/widgets/{slug}.js')]
        if 'class="code-editor"' in body:
            extra_head += [f'<link rel="stylesheet" href="{CODEMIRROR}codemirror.min.css">',
                           f'<link rel="stylesheet" href="{CODEMIRROR}theme/material-darker.min.css">']
            scripts += [f'<script src="{CODEMIRROR}codemirror.min.js"></script>',
                        f'<script src="{CODEMIRROR}mode/python/python.min.js"></script>',
                        f'<script>window.PYODIDE_URL = "{PYODIDE}";</script>',
                        script('assets/terminal.js')]

        prev_ch = chapters[i - 1] if i else None
        next_ch = chapters[i + 1] if i + 1 < len(chapters) else None
        prev_html = (f'<a class="pager prev" href="{prev_ch["slug"]}.html" rel="prev"><span class="pager-dir">← Previous</span>'
                     f'<span class="pager-title">{html.escape(prev_ch["title"])}</span></a>') if prev_ch else '<span></span>'
        next_html = (f'<a class="pager next" href="{next_ch["slug"]}.html" rel="next"><span class="pager-dir">Next →</span>'
                     f'<span class="pager-title">{html.escape(next_ch["title"])}</span></a>') if next_ch else \
            '<a class="pager next" href="certificate.html"><span class="pager-dir">You made it →</span><span class="pager-title">Claim your certificate</span></a>'

        page = render(page_tpl, {
            **frame(ch['title'], ch['hook'], slug, ch, extra_head),
            'number': ch['number'],
            'part_id': part['id'],
            'part_number': part['number'],
            'part_title': html.escape(part['title']),
            'part_color': part['color'],
            'part_tint': part['tint'],
            'hook': html.escape(ch['hook']),
            'minutes': max(2, round(len(plain_text(body).split()) / 180)),
            'body': body,
            'prev': prev_html,
            'next': next_html,
            'scripts': '\n'.join(scripts),
        })
        (out / f'{slug}.html').write_text(page)

    if missing and not args.draft:
        sys.exit('missing chapters: ' + ', '.join(missing) + ' (use --draft to build anyway)')

    manifest = {
        'chapters': [{'slug': c['slug'], 'title': c['title'], 'hook': c['hook'], 'part': c['part']['id'], 'number': c['number']} for c in chapters],
        'parts': [{'id': p['id'], 'number': p['number'], 'title': p['title'], 'color': p['color'], 'tint': p['tint']} for p in book['parts']],
    }
    (out / 'assets' / 'book-data.js').write_text(
        'window.BOOK = ' + json.dumps(manifest) + ';\nwindow.GLOSSARY = ' + json.dumps(glossary, ensure_ascii=False) + ';\n')
    (out / 'assets' / 'search-index.json').write_text(json.dumps(search, ensure_ascii=False))

    part_cards = []
    for part in book['parts']:
        items = ''.join(
            f'<li><a href="{c["slug"]}.html" data-slug="{c["slug"]}"><span class="card-num">{c["number"]}</span>'
            f'<span class="card-text"><strong>{html.escape(c["title"])}</strong><em>{html.escape(c["hook"])}</em></span></a></li>'
            for c in part['chapters'])
        part_cards.append(
            f'<section class="part-card" id="part-{part["id"]}" style="--part:{part["color"]};--part-tint:{part["tint"]}">'
            f'<header><span class="part-kicker">Part {part["number"]}</span><h2>{html.escape(part["title"])}</h2>'
            f'<p>{html.escape(part["blurb"])}</p>'
            f'<div class="part-meter"><span data-part-bar="{part["id"]}"></span></div></header><ol>{items}</ol></section>')

    (out / 'index.html').write_text(render((SRC / 'home.html').read_text(), {
        **frame(book['tagline'], book['description'], 'index'),
        'parts': '\n'.join(part_cards),
        'widgets': widgets,
        'scripts_count': len(list(PY.glob('*.py'))),
        'first': chapters[0]['slug'],
        'scripts': '\n'.join(base_scripts + ([script('assets/home.js')] if (SRC / 'assets' / 'home.js').exists() else [])),
    }))

    terms = ''.join(f'<dt id="{html.escape(t.lower().replace(" ", "-"))}">{html.escape(t)}</dt><dd>{html.escape(d)}</dd>'
                    for t, d in sorted(glossary.items(), key=lambda kv: kv[0].lower()))
    simple_pages = [
        ('glossary', 'Glossary', 'Every special word in the book, in plain language.',
         '<p class="lede">Every special word in the book, in plain language. In the chapters, words with a dotted '
         f'underline show these same definitions when you hover or tap.</p><dl class="glossary-list">{terms}</dl>', []),
        ('404', 'Page not found', 'That page is not in the book.',
         '<p class="lede">That page is not in the book. Try the <a href="index.html">contents</a>.</p>', []),
    ]
    for page in book.get('pages', []):
        src = SRC / 'pages' / f'{page["slug"]}.html'
        if not src.exists():
            if not args.draft:
                sys.exit(f'missing page: {src}')
            continue
        extra = [script(f'assets/{page["slug"]}.js')] if (SRC / 'assets' / f'{page["slug"]}.js').exists() else []
        simple_pages.append((page['slug'], page['title'], page['description'],
                             expand_refs(src.read_text(), by_slug, src.name), extra))
    for slug, title, description, body, extra in simple_pages:
        (out / f'{slug}.html').write_text(render(simple_tpl, {
            **frame(title, description, slug),
            'body': body,
            'scripts': '\n'.join(base_scripts + extra),
        }))

    note = f' ({len(missing)} not written yet: {", ".join(missing)})' if missing else ''
    print(f'built {len(chapters)} chapters, {widgets} interactive widgets -> {out}{note}')


if __name__ == '__main__':
    main()
