"""
test_compression.py

Unit tests for Variable Byte and Delta encoding.
"""

import unittest
from compression import (
    vbyte_encode_number,
    vbyte_decode,
    delta_encode,
    delta_decode,
    compress_postings_list,
    decompress_postings_list,
)


class TestCompression(unittest.TestCase):
    def test_single_number_encode_decode(self):
        for n in [0, 1, 127, 128, 255, 1024, 16384, 1000000]:
            encoded = vbyte_encode_number(n)
            decoded = vbyte_decode(encoded)
            self.assertEqual(decoded, [n])

    def test_multi_number_stream(self):
        numbers = [5, 130, 256, 10000, 42]
        stream = bytearray()
        for num in numbers:
            stream.extend(vbyte_encode_number(num))
        decoded = vbyte_decode(bytes(stream))
        self.assertEqual(decoded, numbers)

    def test_delta_encoding(self):
        doc_ids = [100, 105, 112, 120, 250]
        deltas = delta_encode(doc_ids)
        self.assertEqual(deltas, [100, 5, 7, 8, 130])
        reconstructed = delta_decode(deltas)
        self.assertEqual(reconstructed, doc_ids)

    def test_postings_roundtrip(self):
        postings = {
            1: [0, 4, 12],
            5: [2, 15],
            12: [1, 3, 5, 7, 9],
        }
        compressed = compress_postings_list(postings)
        decompressed = decompress_postings_list(compressed)
        self.assertEqual(decompressed, postings)

    def test_compression_savings(self):
        # 3 documents with 5 positions each = 1 + 3*(1 + 1 + 5) = 22 integers
        # In uncompressed 32-bit (4-byte) pack, that's 22 * 4 = 88 bytes.
        postings = {
            100: [1, 2, 3, 4, 5],
            105: [10, 12, 14, 16, 18],
            110: [20, 25, 30, 35, 40],
        }
        compressed = compress_postings_list(postings)
        # In VByte with deltas, small numbers (<= 127) take only 1 byte each!
        raw_size = (1 + 3 * (2 + 5)) * 4  # 88 bytes
        self.assertLess(len(compressed), raw_size / 2)


if __name__ == "__main__":
    unittest.main()

