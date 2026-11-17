#!/usr/bin/env python3
"""
Build py/names.txt and py/dinosaurs.txt, the word lists for the "names" chapter.

    curl -L -o names_raw.txt https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
    curl -L -o dinosaurs_raw.txt "https://en.wikipedia.org/w/index.php?title=List_of_dinosaur_genera&action=raw"
    python3 tools/make_names.py names_raw.txt dinosaurs_raw.txt py

names_raw.txt is the makemore list of US baby names from 2018 (public-domain
Social Security data). It is two lists back to back, girls then boys, each
most-popular-first; the boys' list begins at the second "liam". We keep the
4,300 most popular of each, minus two-letter entries (mostly initials: cj, jr).

dinosaurs_raw.txt is the wikitext of Wikipedia's "List of dinosaur genera"
(CC BY-SA 4.0). We keep the formally published names that the list gives
without a caveat, which drops synonyms, informal names and fossils that
turned out not to be dinosaurs.

Both outputs are lowercase a-z, one word per line, with no repeats, and are
shuffled with a fixed seed so a reader of the file can use the last tenth as
a held-out set.
"""
import random
import re
import sys
from pathlib import Path

PER_LIST = 4300
NOT_FOR_KIDS = re.compile(
    r"^(unknown|lucifer|riot|demoni|goddess|passion|erectopus)$|"
    r"fuck|shit|piss|bitch|slut|whore|porn|penis|vagin|nazi|hitler|rape|nigg"
)
ACCEPTED_GENUS = re.compile(r"^\*\s*''\[\[(?:[^\]|]*\|)?([^\]|]+)\]\]''\s*(?:<ref.*)?$")


def keep(words, shortest):
    seen = dict.fromkeys(w.lower() for w in words)
    out = [w for w in seen if re.fullmatch(r"[a-z]+", w) and len(w) >= shortest and not NOT_FOR_KIDS.search(w)]
    random.Random(13).shuffle(out)
    return out


def first_names(path):
    names = path.read_text(encoding="utf-8").split()
    boys_start = names.index("liam", names.index("liam") + 1)
    return keep(names[:PER_LIST] + names[boys_start:boys_start + PER_LIST], shortest=3)


def dinosaurs(path):
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ACCEPTED_GENUS.match(line.strip())
        if m:
            found.append(m.group(1))
    return keep(found, shortest=2)


def main(names_src, dinosaurs_src, out_dir):
    for filename, words in [
        ("names.txt", first_names(Path(names_src))),
        ("dinosaurs.txt", dinosaurs(Path(dinosaurs_src))),
    ]:
        (Path(out_dir) / filename).write_text("\n".join(words) + "\n", encoding="utf-8")
        print(f"wrote {len(words)} words to {out_dir}/{filename}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
