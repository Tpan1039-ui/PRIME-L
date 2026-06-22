from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Iterable, Sequence


CANONICAL_AAS = tuple("ACDEFGHIKLMNPQRSTVWY")
MUT_RE = re.compile(r"([A-Z])(\d+)([A-Z])")
WT_PREFIXES = ("WT", "WILD")


@dataclass(frozen=True, order=True)
class Mutation:
    wt: str
    pos: int
    mt: str

    @property
    def one_based(self) -> int:
        return self.pos + 1

    def label(self) -> str:
        return f"{self.wt}{self.one_based}{self.mt}"


def is_wt_label(label: object) -> bool:
    text = str(label).strip().upper()
    return not text or any(text.startswith(prefix) for prefix in WT_PREFIXES)


def parse_mutations(label: object, wild_type: str | None = None) -> tuple[Mutation, ...]:
    if is_wt_label(label):
        return tuple()
    text = str(label).strip().replace("/", ":").replace(",", ":").replace(";", ":")
    matches = MUT_RE.findall(text)
    if not matches:
        raise ValueError(f"Cannot parse mutation label: {label!r}")
    muts = []
    seen: dict[int, Mutation] = {}
    for wt, pos_text, mt in matches:
        pos = int(pos_text) - 1
        if wild_type is not None:
            if pos < 0 or pos >= len(wild_type):
                raise ValueError(f"Position out of range in {label!r}")
            if wild_type[pos] != wt:
                raise ValueError(
                    f"Wild-type mismatch in {label!r}: expected {wild_type[pos]} at {pos + 1}, got {wt}"
                )
        mut = Mutation(wt, pos, mt)
        if pos in seen and seen[pos] != mut:
            raise ValueError(f"Conflicting mutations at position {pos + 1}: {label!r}")
        seen[pos] = mut
        muts.append(mut)
    return tuple(sorted(muts, key=lambda m: m.pos))


def format_mutations(mutations: Sequence[Mutation]) -> str:
    return "WT" if not mutations else ":".join(mut.label() for mut in sorted(mutations, key=lambda m: m.pos))


def apply_mutations(wild_type: str, mutations: Sequence[Mutation]) -> str:
    seq = list(wild_type)
    for mut in mutations:
        if seq[mut.pos] != mut.wt:
            raise ValueError(f"Wild-type mismatch at {mut.one_based}: expected {seq[mut.pos]}, got {mut.wt}")
        seq[mut.pos] = mut.mt
    return "".join(seq)


def enumerate_single_mutants(wild_type: str, amino_acids: Iterable[str] = CANONICAL_AAS) -> list[str]:
    labels = []
    for pos, wt in enumerate(wild_type):
        if wt not in CANONICAL_AAS:
            continue
        for mt in amino_acids:
            if mt != wt:
                labels.append(Mutation(wt, pos, mt).label())
    return labels


def unique_single_mutations(labels: Iterable[str], wild_type: str) -> list[Mutation]:
    singles: dict[tuple[int, str], Mutation] = {}
    for label in labels:
        for mut in parse_mutations(label, wild_type):
            singles[(mut.pos, mut.mt)] = mut
    return sorted(singles.values(), key=lambda m: (m.pos, m.mt))


def combine_single_mutations(
    seeds: Sequence[Mutation],
    min_sites: int,
    max_sites: int,
    exclude: Iterable[str] = (),
    max_candidates: int | None = None,
) -> list[str]:
    excluded = set(exclude)
    candidates: list[str] = []
    for size in range(min_sites, max_sites + 1):
        for combo in itertools.combinations(seeds, size):
            positions = [mut.pos for mut in combo]
            if len(set(positions)) != len(positions):
                continue
            label = format_mutations(combo)
            if label in excluded:
                continue
            candidates.append(label)
            if max_candidates is not None and len(candidates) >= max_candidates:
                return candidates
    return candidates
