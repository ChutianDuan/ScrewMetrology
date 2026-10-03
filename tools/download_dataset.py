#!/usr/bin/env python3
"""Download and verify the MVTec Screws V1.0 archive; Python standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

URL = 'https://osf.io/ruca6/download'
SHA256 = 'a742c5744cea973eca5d9eeee5896c8b458d3d695fb107a58707a9e7e0cc1342'
OFFICIAL = 'https://www.mvtec.com/research-teaching/datasets/mvtec-screws'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    archive = args.data_dir / 'mvtec_screws_v1.0.tar.gz'
    if not archive.exists():
        partial = archive.with_suffix(archive.suffix + '.part')
        print('Downloading approximately 1.3 GB from', URL, flush=True)
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open('wb') as out:
            while chunk := response.read(4 * 1024 * 1024):
                out.write(chunk)
        if digest(partial) != SHA256:
            raise RuntimeError('Archive SHA-256 mismatch; partial file retained for inspection')
        partial.replace(archive)
    if digest(archive) != SHA256:
        raise RuntimeError('Existing archive SHA-256 mismatch; do not extract it')
    root = (args.data_dir / 'mvtec_screws').resolve()
    root.mkdir(exist_ok=True)
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root) or not (member.isfile() or member.isdir()):
                raise RuntimeError('Unsafe archive entry: ' + member.name)
            stream.extract(member, root)
    metadata = json.loads((root / 'mvtec_screws.json').read_text())
    if len(metadata['images']) != 384 or len(metadata['categories']) != 13:
        raise RuntimeError('Unexpected dataset inventory')
    names = [image['file_name'] for image in metadata['images']]
    ids = [annotation['id'] for annotation in metadata['annotations']]
    if len(names) != len(set(names)) or len(ids) != len(set(ids)):
        raise RuntimeError('Duplicate image names or annotation IDs')
    if any(not (root / 'images' / name).is_file() for name in names):
        raise RuntimeError('Missing dataset image')
    source = {'official_page': OFFICIAL, 'mirror': URL, 'archive_sha256': SHA256,
              'license': 'CC BY-NC-SA 4.0', 'images': len(names),
              'annotations_in_archive': len(ids), 'annotations_on_official_page': 4426,
              'note': 'The downloaded original archive contains 4427 unique annotation IDs.'}
    (root / 'SOURCE.json').write_text(json.dumps(source, indent=2) + '\n')
    print('Verified:', source['images'], 'images,', source['annotations_in_archive'], 'annotations')


if __name__ == '__main__':
    main()
