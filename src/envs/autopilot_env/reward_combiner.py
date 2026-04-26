"""
Single point of composition for all reward terms.

Each term is computed independently and combined with explicit weights.
The `mode` field exists for ablation toggling — flipping it does not require
restarting the environment.

Modes:
  full          : extrinsic + pbrs + intrinsic   (default)
  proxy_only    : extrinsic only                  (baseline)
  no_pbrs       : extrinsic + intrinsic
  no_intrinsic  : extrinsic + pbrs
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RewardComponents:
    """
    Canonical reward decomposition used across the environment, diagnostics,
    frontend, and training logs.

    The current v2 runtime only produces `extrinsic`, `pbrs_shaping`, and
    `intrinsic_count`. The remaining fields are kept at zero for now so later
    phases can wire them in without changing the schema again.
    """
    extrinsic: float = 0.0
    pbrs_shaping: float = 0.0
    intrinsic_count: float = 0.0
    intrinsic_rnd: float = 0.0
    weighted_judge: float = 0.0
    difference_reward: float = 0.0
    ird_posterior_correction: float = 0.0


@dataclass
class RewardCombiner:
    mode: str               = "full"
    w_extrinsic: float      = 1.0
    w_pbrs: float           = 1.0
    w_intrinsic: float      = 1.0
    w_intrinsic_rnd: float  = 1.0
    w_judge: float          = 1.0
    w_difference: float     = 0.0
    w_ird: float            = 1.0

    def _gates(self) -> Dict[str, float]:
        if self.mode == "proxy_only":
            return {
                "extrinsic": self.w_extrinsic,
                "pbrs_shaping": 0.0,
                "intrinsic_count": 0.0,
                "intrinsic_rnd": 0.0,
                "weighted_judge": 0.0,
                "difference_reward": 0.0,
                "ird_posterior_correction": 0.0,
            }
        if self.mode == "no_pbrs":
            return {
                "extrinsic": self.w_extrinsic,
                "pbrs_shaping": 0.0,
                "intrinsic_count": self.w_intrinsic,
                "intrinsic_rnd": self.w_intrinsic_rnd,
                "weighted_judge": self.w_judge,
                "difference_reward": self.w_difference,
                "ird_posterior_correction": self.w_ird,
            }
        if self.mode == "no_intrinsic":
            return {
                "extrinsic": self.w_extrinsic,
                "pbrs_shaping": self.w_pbrs,
                "intrinsic_count": 0.0,
                "intrinsic_rnd": 0.0,
                "weighted_judge": self.w_judge,
                "difference_reward": self.w_difference,
                "ird_posterior_correction": self.w_ird,
            }
        return {
            "extrinsic": self.w_extrinsic,
            "pbrs_shaping": self.w_pbrs,
            "intrinsic_count": self.w_intrinsic,
            "intrinsic_rnd": self.w_intrinsic_rnd,
            "weighted_judge": self.w_judge,
            "difference_reward": self.w_difference,
            "ird_posterior_correction": self.w_ird,
        }

    def combine(
        self,
        components: Optional[RewardComponents] = None,
        *,
        extrinsic: float = 0.0,
        pbrs: float = 0.0,
        intrinsic: float = 0.0,
    ) -> Dict[str, float]:
        """
        Return a fully-decomposed reward dict including the weighted total.

        The keyword-only scalar arguments are kept for backward compatibility
        with the current v2 call sites.
        """
        if components is None:
            components = RewardComponents(
                extrinsic=extrinsic,
                pbrs_shaping=pbrs,
                intrinsic_count=intrinsic,
            )

        g = self._gates()
        weighted = {
            "extrinsic": g["extrinsic"] * components.extrinsic,
            "pbrs_shaping": g["pbrs_shaping"] * components.pbrs_shaping,
            "intrinsic_count": g["intrinsic_count"] * components.intrinsic_count,
            "intrinsic_rnd": g["intrinsic_rnd"] * components.intrinsic_rnd,
            "weighted_judge": g["weighted_judge"] * components.weighted_judge,
            "difference_reward": g["difference_reward"] * components.difference_reward,
            "ird_posterior_correction": g["ird_posterior_correction"] * components.ird_posterior_correction,
        }
        weighted["total"] = round(sum(weighted.values()), 4)
        weighted["mode"] = self.mode
        return weighted
