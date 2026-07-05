"""Workflow strategy dataclass and registry mapping --strategy values to the four pipeline stages."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict


@dataclass(frozen=True)
class WorkflowStrategy:
    expression_matrix_type: str  # directLFQ | TMT-Integrator abundance | reporter intensity
    normalization: str           # none (all 7 strategies use "no extra normalization")
    imputation: str              # SeqKNN | Impseq | MinDet | MinProb | bpca
    dea_method: str              # DEqMS | limma | ROTS | proDA

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


STRATEGY_REGISTRY: Dict[str, WorkflowStrategy] = {
    "fragpipe-dda":    WorkflowStrategy("directLFQ",                "none", "SeqKNN",  "DEqMS"),
    "maxquant-dda":    WorkflowStrategy("directLFQ",                "none", "Impseq",  "DEqMS"),
    "diann-dia":       WorkflowStrategy("directLFQ",                "none", "MinDet",  "limma"),
    "spectronaut-dia": WorkflowStrategy("directLFQ",                "none", "Impseq",  "ROTS"),
    "fragpipe-tmt":    WorkflowStrategy("TMT-Integrator abundance", "none", "SeqKNN",  "limma"),
    "maxquant-tmt":    WorkflowStrategy("reporter intensity",       "none", "bpca",    "proDA"),
    "generic":         WorkflowStrategy("directLFQ",                "none", "MinProb", "limma"),
}


def resolve_strategy(name: str) -> WorkflowStrategy:
    if name not in STRATEGY_REGISTRY:
        valid = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(f"Unknown --strategy '{name}'. Valid values: {valid}")
    return STRATEGY_REGISTRY[name]