import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import DelayedPDActuatorCfg


# Configuration for Klask articulation
s2r = False
if s2r:
    KLASK_PARAMS = {
        "decimation": 20,
        "physics_dt": 0.001,
        "timeout": 10.0,
        "actuator_delay": (0.0, 0.0),
        "action_history": 10,
        "player_goal": (0.0, -0.176215, 0.01905),
        "opponent_goal": (0.0, 0.176215, 0.01905),
        "ball_restitution": 0.3,
        "ball_static_friction": 0.3,
        "ball_dynamic_friction": 0.6,
        "ball_mass_initial": 0.002,
        "ball_reset_position_x": (-0.15, 0.15),
        "ball_reset_position_y": (-0.21, 0.21),
        "max_ball_vel": 5.0,  # maximum speed the ball may have for a goal to be counted
        "domain_randomization": {
            "use_domain_randomization": True,
            "static_friction_range": (0.1, 0.5),
            "dynamic_friction_range": (0.2, 0.4),
            "restitution_range": (0.25, 0.4),
            "ball_mass_range": (0.003, 0.006),
            "stiffness_range": (0.0, 0.0),
            "damping_range": (10.0, 10.0),
        },
    }

else:
    KLASK_PARAMS = {
        "decimation": 20,  # system is running at 50Hz (night shift with 100Hz)
        "physics_dt": 0.001,
        "timeout": 5.0,
        "actuator_delay": (0.0, 0.0),
        "action_history": 0,
        "player_goal": (0.0, -0.176215, 0.01905),
        "opponent_goal": (0.0, 0.176215, 0.01905),
        "ball_restitution": 0.8,
        "ball_static_friction": 0.03,
        "ball_dynamic_friction": 0.03,
        "ball_mass_initial": 0.002,
        "ball_reset_position_x": (-0.15, 0.15),
        "ball_reset_position_y": (-0.21, 0.21),
        "max_ball_vel": 100.0,  # maximum speed the ball may have for a goal to be counted
        "edge": (-0.16, 0.16, -0.22, -0.02),
        "additional_observations": False,
        "domain_randomization": {
            "use_domain_randomization": False,
            "static_friction_range": (0.2, 0.5),
            "dynamic_friction_range": (0.3, 0.45),
            "restitution_range": (0.95, 0.99),
            "ball_mass_range": (0.0018, 0.022),
            "stiffness_range": (0.0, 0.0),
            "damping_range": (10.0, 10.0),
        },
    }

KLASK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(
            os.path.dirname(__file__), "klask.usd"
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "ground_to_slider_1": -0.115,  # Middle of [-0.210, -0.020] range
            "ground_to_slider_2": 0.115,  # Middle of [0.020, 0.210] range
            "slider_to_peg_1": 0.0,
            "slider_to_peg_2": 0.0,
        }
    ),
    actuators={
        "peg_1x_actuator": DelayedPDActuatorCfg(
            joint_names_expr=["slider_to_peg_1"],
            stiffness=0.0,
            damping=10.0,
            velocity_limit=3.0,
            effort_limit=30.0,
            min_delay=int(
                KLASK_PARAMS["actuator_delay"][0] / KLASK_PARAMS["physics_dt"]
            ),
            max_delay=int(
                KLASK_PARAMS["actuator_delay"][1] / KLASK_PARAMS["physics_dt"]
            ),
        ),
        "peg_1y_actuator": DelayedPDActuatorCfg(
            joint_names_expr=["ground_to_slider_1"],
            stiffness=0.0,
            damping=100.0,
            velocity_limit=3.0,
            effort_limit=300.0,
            min_delay=int(
                KLASK_PARAMS["actuator_delay"][0] / KLASK_PARAMS["physics_dt"]
            ),
            max_delay=int(
                KLASK_PARAMS["actuator_delay"][1] / KLASK_PARAMS["physics_dt"]
            ),
        ),
        "peg_2x_actuator": DelayedPDActuatorCfg(
            joint_names_expr=["slider_to_peg_2"],
            stiffness=0.0,
            damping=10.0,
            velocity_limit=3.0,
            effort_limit=30.0,
            min_delay=int(
                KLASK_PARAMS["actuator_delay"][0] / KLASK_PARAMS["physics_dt"]
            ),
            max_delay=int(
                KLASK_PARAMS["actuator_delay"][1] / KLASK_PARAMS["physics_dt"]
            ),
        ),
        "peg_2y_actuator": DelayedPDActuatorCfg(
            joint_names_expr=["ground_to_slider_2"],
            stiffness=0.0,
            damping=100.0,
            velocity_limit=3.0,
            effort_limit=300.0,
            min_delay=int(
                KLASK_PARAMS["actuator_delay"][0] / KLASK_PARAMS["physics_dt"]
            ),
            max_delay=int(
                KLASK_PARAMS["actuator_delay"][1] / KLASK_PARAMS["physics_dt"]
            ),
        ),
    },
)
