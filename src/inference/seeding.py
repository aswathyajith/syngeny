import hashlib


def derive_seed(seed: int, temperature: float, word: str) -> int:
    """Seed for one (run seed, temperature, word) generation.

    Mixing in the temperature and word keeps sampling noise independent across them, so
    similarity between temperatures or words reflects the output distributions and not
    shared random draws. Deterministic, so resumed runs reproduce the same outputs.
    """

    key = f"{seed}|{temperature:g}|{word}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")  # fits the 32-bit range numpy/torch accept
