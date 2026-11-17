#!/usr/bin/env python3
"""
Build headlines.csv (10,000 rows: topic,text) from the AG News training set.

    curl -LO https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv
    python3 tools/make_headlines.py train.csv headlines.csv

The sample is balanced (2,500 per topic), pre-shuffled with a fixed seed, and
skips stories whose wording is too grim for a book aimed at 13-year-olds.
"""
import csv
import random
import re
import sys

TOPICS = {"1": "world", "2": "sports", "3": "business", "4": "scitech"}
PER_TOPIC = 2500
MAX_CHARS = 220

SKIP = re.compile(
    r"\b(kill\w*|murder\w*|rape\w*|sex\w*|porn\w*|suicid\w*|behead\w*|massacre\w*|"
    r"dead|death\w*|dies|died|bomb\w*|hostage\w*|slain|shot|shooting\w*|stab\w*|"
    r"tortur\w*|abus\w*|molest\w*|genocide|corpse\w*|execut\w*|terror\w*|drug\w*|"
    r"viagra|gay|nude|naked|assault\w*)\b",
    re.I,
)
# A trailing "(Reuters)" or a leading "AP - " names the news agency, and some
# agencies only cover one topic. Left in, the model learns the agency instead.
SOURCE_TAG = re.compile(r"\s*\([^)]*\)\s*$")
SOURCE_LEAD = re.compile(r"^[A-Za-z. ]{2,30}\s-\s")


def clean(title, blurb):
    title = SOURCE_TAG.sub("", title.replace("\\", " "))
    blurb = SOURCE_LEAD.sub("", blurb.replace("\\", " "))
    text = re.sub(r"\s+", " ", f"{title}. {blurb}").strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rsplit(" ", 1)[0]
    return text


def main(src, dst):
    rng = random.Random(13)
    rows = list(csv.reader(open(src, encoding="utf-8")))
    rng.shuffle(rows)
    picked = {t: [] for t in TOPICS.values()}
    for label, title, blurb in rows:
        topic = TOPICS[label]
        text = clean(title, blurb)
        if len(picked[topic]) >= PER_TOPIC or SKIP.search(text):
            continue
        if not text.isascii() or re.search(r"[#&<>;\"]|http|quot|www", text):
            continue
        picked[topic].append((topic, text))
    out = [row for group in picked.values() for row in group]
    rng.shuffle(out)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        w.writerow(["topic", "text"])
        w.writerows(out)
    print(f"wrote {len(out)} rows to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
