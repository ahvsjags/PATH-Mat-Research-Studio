"""Range-extract only intervention UV-Vis CSVs from the public Zenodo ZIP.

The 1.17-GB archive is not downloaded wholesale. The script retrieves the ZIP
central directory and then byte ranges only for the intervention CSV entries
and their experiment metadata file. Each extracted member is CRC/size checked.
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
import time
import zlib
from pathlib import Path, PurePosixPath

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "zenodo_8303993"
BASE_URL = "https://zenodo.org/records/8303993/files/Data_Share%20_V2_2023_08_30.zip"
OFFICIAL_SIZE = 1_173_752_417
OFFICIAL_MD5 = "886fad0d8f439a7602b7ce9cd928c2f3"
TIMEOUT = 45
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "PATH-Mat-reproducible-public-data-audit/1.0"})


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_range(start: int, end: int, total: int | None = None) -> tuple[bytes, dict]:
    last_error = None
    for attempt in range(5):
        try:
            r = SESSION.get(BASE_URL, headers={"Range": f"bytes={start}-{end}"}, timeout=TIMEOUT, stream=True)
            # Fail before reading if the server/proxy ignored Range; otherwise a
            # 200 response could stream the entire 1.17-GB archive.
            if r.status_code != 206:
                r.close()
                raise RuntimeError(f"Range request {start}-{end}: expected 206, got {r.status_code}")
            body = b"".join(r.iter_content(256 * 1024))
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))
    else:
        raise RuntimeError(f"Range request failed: {last_error}")
    if r.status_code != 206:
        raise RuntimeError(f"Range request {start}-{end}: expected 206, got {r.status_code}")
    cr = r.headers.get("Content-Range", "")
    m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", cr)
    if not m:
        raise RuntimeError(f"Missing/invalid Content-Range: {cr!r}")
    a, b, n = map(int, m.groups())
    if (a, b) != (start, end) or len(body) != end - start + 1:
        raise RuntimeError(f"Range mismatch: got {cr}, {len(body)} bytes")
    if total is not None and n != total:
        raise RuntimeError(f"Archive size changed: expected {total}, got {n}")
    return body, {"http_status": r.status_code, "content_range": cr, "etag": r.headers.get("ETag")}


def parse_eocd(tail: bytes, tail_start: int):
    sig = b"PK\x05\x06"
    pos = tail.rfind(sig)
    if pos < 0 or pos + 22 > len(tail):
        raise RuntimeError("ZIP End-of-Central-Directory record not found")
    vals = struct.unpack_from("<4s4H2LH", tail, pos)
    _, disk, cd_disk, n_disk, n_total, cd_size, cd_off, comment_len = vals
    if disk or cd_disk or n_disk != n_total:
        raise RuntimeError("Multi-disk ZIP is not supported")
    if n_total == 0xFFFF or cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF:
        raise RuntimeError("ZIP64 directory needs ZIP64 EOCD handling")
    if pos + 22 + comment_len != len(tail):
        # The last EOCD signature in a byte sequence can be inside a comment.
        raise RuntimeError("EOCD comment length does not terminate at archive end")
    return {"entry_count": n_total, "central_directory_size": cd_size,
            "central_directory_offset": cd_off, "eocd_offset": tail_start + pos}


def parse_central_directory(data: bytes, expected_count: int):
    entries = []
    p = 0
    while p < len(data):
        if data[p:p+4] != b"PK\x01\x02":
            raise RuntimeError(f"Bad central-directory signature at byte {p}")
        fixed = struct.unpack_from("<4s6H3L5H2L", data, p)
        (_, made, needed, flags, method, mtime, mdate, crc, csize, usize,
         name_len, extra_len, comment_len, disk, int_attr, ext_attr, local_off) = fixed
        end = p + 46 + name_len + extra_len + comment_len
        name_b = data[p+46:p+46+name_len]
        name = name_b.decode("utf-8" if flags & 0x800 else "cp437")
        extra = data[p+46+name_len:p+46+name_len+extra_len]
        # Resolve ZIP64 sizes/offset only if needed (not expected for selected small CSVs).
        if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF or local_off == 0xFFFFFFFF:
            q = 0
            while q + 4 <= len(extra):
                tag, size = struct.unpack_from("<HH", extra, q)
                payload = extra[q+4:q+4+size]
                if tag == 0x0001:
                    z = 0
                    if usize == 0xFFFFFFFF: usize = struct.unpack_from("<Q", payload, z)[0]; z += 8
                    if csize == 0xFFFFFFFF: csize = struct.unpack_from("<Q", payload, z)[0]; z += 8
                    if local_off == 0xFFFFFFFF: local_off = struct.unpack_from("<Q", payload, z)[0]
                    break
                q += 4 + size
        entries.append({"name": name, "flags": flags, "method": method, "crc32": crc,
                        "compressed_size": csize, "uncompressed_size": usize,
                        "local_header_offset": local_off, "disk_start": disk})
        p = end
    if p != len(data) or len(entries) != expected_count:
        raise RuntimeError(f"Central directory parse mismatch: {len(entries)} entries, expected {expected_count}")
    return entries


def is_target(name: str) -> bool:
    low = name.replace("\\", "/").casefold()
    return low.endswith("/data/dls/2023_01_27_z2_z2m246i/exported.txt") or low.endswith("/data/dls/2023_01_27_z2_z2m246i/si.csv")


def extract_entry_from_block(entry: dict, block: bytes, block_start: int):
    off = entry["local_header_offset"]
    rel_off = off - block_start
    hdr = block[rel_off:rel_off + 30]
    if len(hdr) != 30:
        raise RuntimeError(f"Local header outside range block for {entry['name']}")
    vals = struct.unpack("<4s5H3L2H", hdr)
    sig, ver, flags, method, mtime, mdate, crc, csize, usize, name_len, extra_len = vals
    if sig != b"PK\x03\x04":
        raise RuntimeError(f"Bad local-header signature for {entry['name']}")
    if method != entry["method"] or crc != entry["crc32"]:
        raise RuntimeError(f"Local/central metadata mismatch for {entry['name']}")
    start = rel_off + 30 + name_len + extra_len
    comp = block[start:start + entry["compressed_size"]]
    if len(comp) != entry["compressed_size"]:
        raise RuntimeError(f"Member payload outside range block for {entry['name']}")
    if method == 0:
        content = comp
    elif method == 8:
        content = zlib.decompress(comp, -15)
    else:
        raise RuntimeError(f"Unsupported compression method {method} for {entry['name']}")
    crc_actual = zlib.crc32(content) & 0xFFFFFFFF
    if len(content) != entry["uncompressed_size"] or crc_actual != entry["crc32"]:
        raise RuntimeError(f"CRC/size mismatch for {entry['name']}")
    rel = PurePosixPath(entry["name"])
    if rel.is_absolute() or ".." in rel.parts:
        raise RuntimeError(f"Unsafe archive member path: {entry['name']}")
    dest = DEST.joinpath(*rel.parts)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return {"archive_member": entry["name"], "compressed_bytes": len(comp),
            "uncompressed_bytes": len(content), "crc32": f"{crc_actual:08x}",
            "sha256": sha256(content), "local_header_offset": off}


def main():
    # ZIP comments are usually absent; 8 KiB keeps this remote read small.
    # Retry with the ZIP maximum comment window if the EOCD is not present.
    tail_len = 8_192
    tail_start = OFFICIAL_SIZE - tail_len
    tail, tail_meta = get_range(tail_start, OFFICIAL_SIZE - 1, OFFICIAL_SIZE)
    try:
        eocd = parse_eocd(tail, tail_start)
    except RuntimeError:
        tail_len = 65_557
        tail_start = OFFICIAL_SIZE - tail_len
        tail, tail_meta = get_range(tail_start, OFFICIAL_SIZE - 1, OFFICIAL_SIZE)
        eocd = parse_eocd(tail, tail_start)
    cd_start = eocd["central_directory_offset"]
    cd_end = cd_start + eocd["central_directory_size"] - 1
    cd, cd_meta = get_range(cd_start, cd_end, OFFICIAL_SIZE)
    entries = parse_central_directory(cd, eocd["entry_count"])
    (ROOT / "analysis" / "ZENODO_8303993_CENTRAL_DIRECTORY_NAMES_V222.txt").write_text(
        "\n".join(e["name"] for e in entries) + "\n", encoding="utf-8")
    selected = [e for e in entries if is_target(e["name"])]
    if not selected:
        hints = [e["name"] for e in entries if "uv" in e["name"].casefold() or "intervention" in e["name"].casefold()]
        raise RuntimeError(f"No intervention UV-Vis CSV entries found; candidate paths: {hints[:30]}")
    # Selected CSVs are adjacent in the archive. Read each clustered span once
    # instead of making one network request per file.
    selected.sort(key=lambda e: e["local_header_offset"])
    groups, current = [], []
    current_end = -1
    for e in selected:
        # A 64-KiB upper bound covers any local extra field on the group's last member.
        approximate_end = e["local_header_offset"] + 30 + len(e["name"].encode("utf-8")) + 65_535 + e["compressed_size"]
        if current and e["local_header_offset"] - current_end > 2_000_000:
            groups.append(current)
            current = []
            current_end = -1
        current.append(e)
        current_end = max(current_end, approximate_end)
    if current:
        groups.append(current)
    extracted = []
    fetched_groups = []
    for group in groups:
        start = group[0]["local_header_offset"]
        last = group[-1]
        end = min(last["local_header_offset"] + 30 + len(last["name"].encode("utf-8")) + 65_535 + last["compressed_size"] - 1,
                  OFFICIAL_SIZE - 1)
        block, meta = get_range(start, end, OFFICIAL_SIZE)
        fetched_groups.append({"members": len(group), "start": start, "end": end,
                               "fetched_bytes": len(block), "http": meta})
        extracted.extend(extract_entry_from_block(e, block, start) for e in group)
    manifest = {
        "source_record": "https://zenodo.org/records/8303993",
        "source_doi": "10.5281/zenodo.8303993",
        "source_archive_name": "Data_Share _V2_2023_08_30.zip",
        "source_archive_size_bytes": OFFICIAL_SIZE,
        "source_archive_md5_as_published_by_Zenodo": OFFICIAL_MD5,
        "source_archive_md5_verified_locally": False,
        "intake_method": "HTTP byte-range retrieval of ZIP tail, central directory, and selected member payloads; full 1.17-GB ZIP not downloaded",
        "http": {"tail": tail_meta, "central_directory": cd_meta},
        "member_range_groups": fetched_groups,
        "zip_index": {"entry_count": eocd["entry_count"], "central_directory_size": eocd["central_directory_size"],
                      "central_directory_offset": cd_start},
        "selected_member_count": len(extracted),
        "selected_members": extracted,
    }
    out = ROOT / "analysis" / "ZENODO_8303993_DLS_INTAKE_MANIFEST_V222.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
