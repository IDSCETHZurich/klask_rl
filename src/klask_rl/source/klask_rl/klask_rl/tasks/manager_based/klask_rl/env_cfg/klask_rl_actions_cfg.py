from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    player_x = mdp.JointVelocityActionCfg(asset_name="klask", joint_names=["slider_to_peg_1"])

    player_y = mdp.JointVelocityActionCfg(asset_name="klask", joint_names=["ground_to_slider_1"])

    opponent_x = mdp.JointVelocityActionCfg(asset_name="klask", joint_names=["slider_to_peg_2"])

    opponent_y = mdp.JointVelocityActionCfg(asset_name="klask", joint_names=["ground_to_slider_2"])

    """ player = mdp.JointVelocityActionCfg(
        asset_name="klask",
        joint_names=["board_to_peg_1"]
    )

    opponent = mdp.JointVelocityActionCfg(
        asset_name="klask",
        joint_names=["board_to_peg_2"]
    ) """


@configclass
class ActionsCfgPlayerOnly:
    """Action specifications for player-only control (opponent is stationary).

    Used for SAC training where opponent doesn't move.
    """

    player_x = mdp.JointVelocityActionCfg(
        asset_name="klask",
        joint_names=["slider_to_peg_1"],
    )

    player_y = mdp.JointVelocityActionCfg(
        asset_name="klask",
        joint_names=["ground_to_slider_1"],
    )
