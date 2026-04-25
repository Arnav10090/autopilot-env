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
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RewardCombiner:
    mode: str               = "full"
    w_extrinsic: float      = 1.0
    w_pbrs: float           = 1.0
    w_intrinsic: float      = 1.0

    def _gates(self) -> Dict[str, float]:
        if self.mode == "proxy_only":
            return {"extrinsic": self.w_extrinsic, "pbrs": 0.0, "intrinsic": 0.0}
        if self.mode == "no_pbrs":
            return {"extrinsic": self.w_extrinsic, "pbrs": 0.0, "intrinsic": self.w_intrinsic}
        if self.mode == "no_intrinsic":
            return {"extrinsic": self.w_extrinsic, "pbrs": self.w_pbrs, "intrinsic": 0.0}
        return {"extrinsic": self.w_extrinsic, "pbrs": self.w_pbrs, "intrinsic": self.w_intrinsic}

    def combine(
        self,
        extrinsic: float,
        pbrs: float,
        intrinsic: float,
    ) -> Dict[str, float]:
        """Return a fully-decomposed reward dict including the weighted total."""
        g = self._gates()
        weighted = {
            "extrinsic":          g["extrinsic"]  * extrinsic,
            "pbrs_shaping":       g["pbrs"]       * pbrs,
            "intrinsic_count":    g["intrinsic"]  * intrinsic,
        }
        weighted["total"] = round(sum(weighted.values()), 4)
        weighted["mode"] = self.mode
        return weighted
