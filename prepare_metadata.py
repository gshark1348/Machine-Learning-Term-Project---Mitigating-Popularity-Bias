"""Extract the two demo metadata files from a local MovieLens 20M ZIP."""
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
with zipfile.ZipFile(sys.argv[1]) as archive:
    (root / 'genome-tags.csv').write_bytes(archive.read('ml-20m/genome-tags.csv'))
    with archive.open('ml-20m/links.csv') as source:
        rows = csv.DictReader(io.TextIOWrapper(source))
        ids = {r['movieId']: 'tt' + r['imdbId'] for r in rows if r['imdbId']}
    (root / 'poster_ids.json').write_text(json.dumps(ids))
