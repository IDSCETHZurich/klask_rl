import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from klask_rl.assets.robots.klask_params import KLASK_PARAMS

KLASK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.path.dirname(__file__), "klask.usd"),
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
            damping=KLASK_PARAMS["actuator_x_damping"],
            velocity_limit=KLASK_PARAMS["actuator_velocity_limit"],
            effort_limit=KLASK_PARAMS["actuator_x_effort_limit"],
            min_delay=int(KLASK_PARAMS["actuator_delay"][0] / KLASK_PARAMS["physics_dt"]),
            max_delay=int(KLASK_PARAMS["actuator_delay"][1] / KLASK_PARAMS["physics_dt"]),
        ),
        "peg_1y_actuator": DelayedPDActuatorCfg(
            joint_names_expr=["ground_to_slider_1"],
            stiffness=0.0,
            damping=KLASK_PARAMS["actuator_y_damping"],
            velocity_limit=KLASK_PARAMS["actuator_velocity_limit"],
            effort_limit=KLASK_PARAMS["actuator_y_effort_limit"],
            min_delay=int(KLASK_PARAMS["actuator_delay"][0] / KLASK_PARAMS["physics_dt"]),
            max_delay=int(KLASK_PARAMS["actuator_delay"][1] / KLASK_PARAMS["physics_dt"]),
        ),
        "peg_2x_actuator": DelayedPDActuatorCfg(
            joint_names_expr=["slider_to_peg_2"],
            stiffness=0.0,
            damping=KLASK_PARAMS["actuator_x_damping"],
            velocity_limit=KLASK_PARAMS["actuator_velocity_limit"],
            effort_limit=KLASK_PARAMS["actuator_x_effort_limit"],
            min_delay=int(KLASK_PARAMS["actuator_delay"][0] / KLASK_PARAMS["physics_dt"]),
            max_delay=int(KLASK_PARAMS["actuator_delay"][1] / KLASK_PARAMS["physics_dt"]),
        ),
        "peg_2y_actuator": DelayedPDActuatorCfg(
            joint_names_expr=["ground_to_slider_2"],
            stiffness=0.0,
            damping=KLASK_PARAMS["actuator_y_damping"],
            velocity_limit=KLASK_PARAMS["actuator_velocity_limit"],
            effort_limit=KLASK_PARAMS["actuator_y_effort_limit"],
            min_delay=int(KLASK_PARAMS["actuator_delay"][0] / KLASK_PARAMS["physics_dt"]),
            max_delay=int(KLASK_PARAMS["actuator_delay"][1] / KLASK_PARAMS["physics_dt"]),
        ),
    },
)
