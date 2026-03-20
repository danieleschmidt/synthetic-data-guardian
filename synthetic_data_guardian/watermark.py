"""LSB watermarking for numeric columns in synthetic data."""
from typing import List, Dict, Any, Tuple


def _embed_bits(value: float, bit0: int, bit1: int) -> float:
    """Embed two bits into the two LSBs of an integer-rounded float."""
    int_val = int(round(value))
    # Clear two LSBs and set to our bits
    int_val = (int_val & ~3) | ((bit1 & 1) << 1) | (bit0 & 1)
    return float(int_val)


def _extract_bits(value: float) -> Tuple[int, int]:
    """Extract the two LSBs from a float value. Returns (bit0, bit1)."""
    iv = int(round(value))
    return (iv & 1, (iv >> 1) & 1)


def embed_watermark(records: List[Dict[str, Any]], watermark: str, column: str = "salary") -> List[Dict[str, Any]]:
    """
    Embed a watermark string into the two LSBs of a numeric column.
    Uses 2-bit LSB steganography — modifies values by at most 3.
    Each record carries 2 bits, so n records can hold n*2 bits (n/4 chars).
    """
    import copy
    result = copy.deepcopy(records)

    # Convert watermark to bits
    wm_bytes = watermark.encode("utf-8")
    bits = []
    for byte in wm_bytes:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)

    # Embed pairs of bits into column (2 bits per record)
    bit_idx = 0
    for record in result:
        if column in record and bit_idx < len(bits):
            b0 = bits[bit_idx] if bit_idx < len(bits) else 0
            b1 = bits[bit_idx + 1] if bit_idx + 1 < len(bits) else 0
            record[column] = _embed_bits(record[column], b0, b1)
            bit_idx += 2

    return result


def extract_watermark(records: List[Dict[str, Any]], length_bytes: int, column: str = "salary") -> str:
    """Extract watermark from two LSBs of a numeric column."""
    total_bits = length_bytes * 8
    # 2 bits per record, so we need ceil(total_bits / 2) records
    records_needed = (total_bits + 1) // 2
    bits = []
    for record in records[:records_needed]:
        if column in record:
            b0, b1 = _extract_bits(record[column])
            bits.append(b0)
            bits.append(b1)

    # Convert bits back to string
    chars = []
    for i in range(0, min(total_bits, len(bits)) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        chars.append(chr(byte))

    return "".join(chars)


def verify_watermark(records: List[Dict[str, Any]], expected_watermark: str, column: str = "salary") -> Dict[str, Any]:
    """Verify that a watermark is present in the records."""
    extracted = extract_watermark(records, len(expected_watermark), column)
    return {
        "verified": extracted == expected_watermark,
        "expected": expected_watermark,
        "extracted": extracted,
    }
