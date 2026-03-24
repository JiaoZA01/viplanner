from __future__ import annotations

from dataclasses import MISSING

import torch
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass


class ForkliftLowLevelAction(ActionTerm):
    """Low-level control for forklift: rear wheel velocity + steering position.

    Action layout (normalized like other IsaacLab action terms):
      [drive_left, drive_right, steer_left, steer_right]
    where each element is typically in [-1, 1] and gets scaled by the underlying terms.
    """

    cfg: "ForkliftLowLevelActionCfg"

    def __init__(self, cfg: "ForkliftLowLevelActionCfg", env):
        # Make sure ActionTerm can resolve the asset from the scene
        if getattr(cfg, "asset_name", MISSING) is MISSING:
            cfg.asset_name = "robot"
        super().__init__(cfg, env)

        # Compose two action terms
        self._drive_term: ActionTerm = cfg.drive_action.class_type(cfg.drive_action, env)
        self._steer_term: ActionTerm = cfg.steer_action.class_type(cfg.steer_action, env)

        self._action_dim = self._drive_term.action_dim + self._steer_term.action_dim

        # Required buffers for ActionTerm abstract interface
        self._raw_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._processed_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        # store raw
        self._raw_actions[:] = actions

        # split: [drive, steer]
        n_drive = self._drive_term.action_dim
        a_drive = actions[:, :n_drive]
        a_steer = actions[:, n_drive:]

        # forward to sub-terms
        self._drive_term.process_actions(a_drive)
        self._steer_term.process_actions(a_steer)

        # store processed (same layout; underlying terms apply scaling)
        self._processed_actions[:] = actions

    def apply_actions(self):
        self._drive_term.apply_actions()
        self._steer_term.apply_actions()


@configclass
class ForkliftLowLevelActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = ForkliftLowLevelAction

    # IMPORTANT: ActionTerm base uses cfg.asset_name to find the asset in the scene
    asset_name: str = "robot"

    drive_action: ActionTermCfg = MISSING
    steer_action: ActionTermCfg = MISSING
