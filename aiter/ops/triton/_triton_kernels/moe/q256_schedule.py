"""GPU-only routing-dependent schedule; preserve original per-XCD assignment."""

import triton
import triton.language as tl


@triton.jit
def make_job_map(Ids, Experts, Counts, Map, STYLE: tl.constexpr):
    xcd = tl.program_id(0)
    pos = tl.arange(0, 128)
    ordinal = xcd + pos * 8
    physical_tiles = tl.load(Counts) // 256
    tokens = tl.load(Counts + 1)
    boundary = physical_tiles - 32
    suffix = tl.arange(0, 32)
    suffix_ids = tl.load(Experts + boundary + suffix, mask=boundary >= 1, other=-1)
    preceding = tl.load(Experts + boundary - 1, mask=boundary >= 1, other=512)
    qualifies = (
        (tokens == 8192)
        & (boundary >= 1)
        & (preceding != 512)
        & (tl.sum((suffix_ids == 512).to(tl.int32), 0) == 32)
    )
    shared_jobs = tl.where((boundary % 256) == 0, 256, 128)
    jobs = tl.where(qualifies, boundary + shared_jobs, physical_tiles)
    valid = ordinal < jobs
    routed = ordinal < boundary
    tile = (ordinal & -64) + (ordinal % 8) * 8 + ((ordinal // 8) % 8)
    tile = tl.where(ordinal < (boundary // 64) * 64, tile, ordinal)
    tile = tl.where(qualifies & ((boundary % 256) != 0) & (tile < 256), tile ^ 1, tile)
    fragments = tl.full((128,), 1, tl.int32)
    for probe in range(1, 8):
        token = (
            tl.load(
                Ids + tile * 256 + probe * 32,
                mask=valid & routed & qualifies,
                other=8192,
            )
            & 0xFFFFFF
        )
        fragments += (token < tokens).to(tl.int32)
    # Fixed startup+per-live-fragment cost, measured independently of expert ID.
    cost = tl.where(
        routed, 410 + fragments * 22, tl.where(shared_jobs == 256, 200, 280)
    )
    key = tl.where(valid, (cost << 16) | (65535 - ordinal), 0).to(tl.uint32)
    ordered = tl.sort(key, descending=True)
    mapped = (65535 - (ordered & 65535)).to(tl.int32)
    mapped = tl.where(ordered != 0, mapped, -1)
    visit = pos // 32
    local_owner = tl.where(visit % 2 == 0, pos % 32, 31 - (pos % 32))
    slot = visit * 256 + local_owner * 8 + xcd
    # Fallback is authoritative linear owner traversal, without changing geometry.
    identity = tl.where(slot < jobs, slot, -1)
    mapped = tl.where(qualifies & (jobs <= 1024), mapped, identity)
    tl.store(Map + slot, mapped)
