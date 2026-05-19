"""Shared configuration collections for matrix multiplication benchmarks."""

# Standard shapes suite covering tall/skinny, short/wide, square, matvec, and outer products.
# Format: (M, K, N) where:
#   - A is M x K
#   - B is K x N
#   - C is M x N
SHAPES: list[tuple[int, int, int]] = [
    (1024, 1, 1024),
    (1024, 1024, 1024),
    (1024, 128, 1024),
    (128, 1024, 1024),
    (1, 1024, 1024),
    (1, 1, 1024),
    (1024, 1024, 1),
]

# Standard static block dimension configurations.
# Format: (bm, bk, bn) where:
#   - bm is block size along M dimension
#   - bk is block size along K dimension
#   - bn is block size along N dimension
BLOCK_CONFIGS: list[tuple[int, int, int]] = [
    (512, 512, 512),
    (128, 128, 128),
]

# Supported data types for benchmarking
DTYPES: list[str] = ["bfloat16", "float32"]
