from __future__ import annotations
from dataclasses import MISSING

import torch
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass


class ForkliftLowLevelAction(ActionTerm):
    """Low-level control for forklift: rear wheel velocity + steering position.

    Action layout: [w_left_rear, w_right_rear, steer_left, steer_right]
    """

    cfg: "ForkliftLowLevelActionCfg"

    def __init__(self, cfg: "ForkliftLowLevelActionCfg", env):
        # IMPORTANT: ActionTerm.__init__ reads cfg.asset_name immediately.
        # Force it to a valid scene key before calling super().
        if getattr(cfg, "asset_name", MISSING) is MISSING:
            cfg.asset_name = "robot"
        super().__init__(cfg, env)

        # Compose two action terms (these already have their own asset_name="robot")
        self._drive_term: ActionTerm = cfg.drive_action.class_type(cfg.drive_action, env)
        self._steer_term: ActionTerm = cfg.steer_action.class_type(cfg.steer_action, env)

        self._action_dim = self._drive_term.action_dim + self._steer_term.action_dim

        # Required by ActionTerm abstract interface
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
        self._raw_actions[:] = actions

        n_drive = self._drive_term.action_dim
        a_drive = actions[:, :n_drive]
        a_steer = actions[:, n_drive:]

        self._drive_term.process_actions(a_drive)
        self._steer_term.process_actions(a_steer)

        self._processed_actions[:] = actions

    def apply_actions(self):
        self._drive_term.apply_actions()
        self._steer_term.apply_actions()


@configclass
class ForkliftLowLevelActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = ForkliftLowLevelAction

    # Still define it here (nice for clarity), but the runtime patch above makes it bulletproof.
    asset_name: str = "robot"
    drive_action: ActionTermCfg = MISSING
    steer_action: ActionTermCfg = MISSING
