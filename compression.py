"""
compression.py

High-performance Variable Byte (VByte) and Delta (Gap) encoding for inverted index compression.
Used by production IR systems (Lucene, Sphinx, Google) to reduce postings file size by 60-80%.
"""

import struct


def vbyte_encode_number(n: int) -> bytes:
    """Encodes a single unsigned integer using Variable Byte (VByte) encoding."""
    if n < 0:
        raise ValueError("VByte encoding only supports non-negative integers")
    out = bytearray()
    while True:
        out.append(n & 0x7F)
        if n < 128:
            break
        n >>= 7
    # Set the continuation/terminating bit on the last byte
    out[0] |= 0x80
    out.reverse()
    return bytes(out)


def vbyte_decode(byte_data: bytes):
    """Decodes a stream of VByte-encoded bytes back into a list of unsigned integers."""
    numbers = []
    n = 0
    for byte in byte_data:
        if byte & 0x80:
            n = (n << 7) | (byte & 0x7F)
            numbers.append(n)
            n = 0
        else:
            n = (n << 7) | (byte & 0x7F)
    return numbers


def delta_encode(sorted_doc_ids: list) -> list:
    """Converts a sorted list of doc IDs into d-gaps (deltas).

    Example: [100, 105, 112] -> [100, 5, 7]
    """
    if not sorted_doc_ids:
        return []
    deltas = [sorted_doc_ids[0]]
    for i in range(1, len(sorted_doc_ids)):
        deltas.append(sorted_doc_ids[i] - sorted_doc_ids[i - 1])
    return deltas


def delta_decode(deltas: list) -> list:
    """Reconstructs original sorted doc IDs from d-gaps.

    Example: [100, 5, 7] -> [100, 105, 112]
    """
    if not deltas:
        return []
    doc_ids = [deltas[0]]
    for i in range(1, len(deltas)):
        doc_ids.append(doc_ids[-1] + deltas[i])
    return doc_ids


def compress_postings_list(doc_ids_and_positions: dict) -> bytes:
    """Compresses a postings list {doc_id: [positions]} using Delta + VByte encoding.

    Format:
    - Number of docs (VByte)
    - Delta-encoded doc_ids (VByte)
    - For each doc:
      - Term frequency (VByte)
      - Delta-encoded positions (VByte)
    """
    sorted_doc_ids = sorted(doc_ids_and_positions.keys())
    if not sorted_doc_ids:
        return vbyte_encode_number(0)

    out = bytearray()
    # 1. Total docs
    out.extend(vbyte_encode_number(len(sorted_doc_ids)))

    # 2. Delta doc IDs
    deltas = delta_encode(sorted_doc_ids)
    for d in deltas:
        out.extend(vbyte_encode_number(d))

    # 3. For each doc, store tf and delta positions
    for doc_id in sorted_doc_ids:
        positions = doc_ids_and_positions[doc_id]
        out.extend(vbyte_encode_number(len(positions)))
        pos_deltas = delta_encode(sorted(positions))
        for p in pos_deltas:
            out.extend(vbyte_encode_number(p))

    return bytes(out)


def decompress_postings_list(byte_stream: bytes) -> dict:
    """Decompresses VByte-encoded postings back into {doc_id: [positions]}."""
    if not byte_stream:
        return {}

    numbers = vbyte_decode(byte_stream)
    if not numbers or numbers[0] == 0:
        return {}

    num_docs = numbers[0]
    idx = 1

    # Extract delta doc IDs
    doc_deltas = numbers[idx : idx + num_docs]
    doc_ids = delta_decode(doc_deltas)
    idx += num_docs

    postings = {}
    for doc_id in doc_ids:
        tf = numbers[idx]
        idx += 1
        pos_deltas = numbers[idx : idx + tf]
        idx += tf
        postings[doc_id] = delta_decode(pos_deltas)

    return postings
