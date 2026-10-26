"""Assemble the public source-index shards and verify the original CSV digest."""
from pathlib import Path
import gzip,hashlib,json

ROOT=Path(__file__).resolve().parent
def main():
    parts=sorted((ROOT/'selector_parts').glob('primary_source_selector_*.csv.gz'))
    assert len(parts)==8
    output=bytearray();header=None
    for n,p in enumerate(parts):
        lines=gzip.decompress(p.read_bytes()).splitlines(keepends=True)
        if header is None:header=lines[0]
        assert lines[0]==header,p.name
        output.extend(b''.join(lines if n==0 else lines[1:]))
    expected=json.loads((ROOT/'expected_checks.json').read_text())['selector_sha256']
    assert hashlib.sha256(output).hexdigest()==expected
    dst=ROOT/'primary_source_selector.csv';dst.write_bytes(output)
    print('Verified source selector:',dst.name,'rows',len(output.splitlines())-1)
if __name__=='__main__':main()
