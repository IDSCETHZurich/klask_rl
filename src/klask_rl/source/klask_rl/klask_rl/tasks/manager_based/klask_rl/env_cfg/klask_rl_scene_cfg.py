import os
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from klask_rl.assets.robots.klask import KLASK_CFG, KLASK_PARAMS


@configclass
class KlaskRlSceneCfg(InteractiveSceneCfg):
    """Configuration for Klask scene."""

    # ground plane
    # ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.007,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=KLASK_PARAMS["ball_mass_initial"]),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=KLASK_PARAMS["ball_restitution"],
                static_friction=KLASK_PARAMS["ball_static_friction"],
                dynamic_friction=KLASK_PARAMS["ball_dynamic_friction"],
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.032)),
    )

    klask = KLASK_CFG.replace(prim_path="{ENV_REGEX_NS}/Klask")


@configclass
class KlaskRlDreamerSceneCfg(InteractiveSceneCfg):
    """Configuration for Klask Dreamer scene."""

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.007,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=KLASK_PARAMS["ball_mass_initial"]),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.3), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=KLASK_PARAMS["ball_restitution"],
                static_friction=KLASK_PARAMS["ball_static_friction"],
                dynamic_friction=KLASK_PARAMS["ball_dynamic_friction"],
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.032)),
    )

    klask = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Klask",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "..", "assets", "robots", "klask_new.usd"
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
                min_delay=int(KLASK_PARAMS["actuator_delay"][0] / KLASK_PARAMS["physics_dt"]),
                max_delay=int(KLASK_PARAMS["actuator_delay"][1] / KLASK_PARAMS["physics_dt"]),
            ),
            "peg_1y_actuator": DelayedPDActuatorCfg(
                joint_names_expr=["ground_to_slider_1"],
                stiffness=0.0,
                damping=100.0,
                velocity_limit=3.0,
                effort_limit=300.0,
                min_delay=int(KLASK_PARAMS["actuator_delay"][0] / KLASK_PARAMS["physics_dt"]),
                max_delay=int(KLASK_PARAMS["actuator_delay"][1] / KLASK_PARAMS["physics_dt"]),
            ),
            "peg_2x_actuator": DelayedPDActuatorCfg(
                joint_names_expr=["slider_to_peg_2"],
                stiffness=0.0,
                damping=10.0,
                velocity_limit=3.0,
                effort_limit=30.0,
                min_delay=int(KLASK_PARAMS["actuator_delay"][0] / KLASK_PARAMS["physics_dt"]),
                max_delay=int(KLASK_PARAMS["actuator_delay"][1] / KLASK_PARAMS["physics_dt"]),
            ),
            "peg_2y_actuator": DelayedPDActuatorCfg(
                joint_names_expr=["ground_to_slider_2"],
                stiffness=0.0,
                damping=100.0,
                velocity_limit=3.0,
                effort_limit=300.0,
                min_delay=int(KLASK_PARAMS["actuator_delay"][0] / KLASK_PARAMS["physics_dt"]),
                max_delay=int(KLASK_PARAMS["actuator_delay"][1] / KLASK_PARAMS["physics_dt"]),
            ),
        },
    )

    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 100.0),
        ),
        width=48,
        height=63,
        data_types=["rgb"],
        update_period=0.0,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.392),  # Tuned to exactly fit the board from the top view
            # pos=(0.0, 0.0, 0.505),  # Adjusted offset for 64x64 cam resolution
            rot=(0.5, -0.5, 0.5, 0.5),
            convention="world",
        ),
    )
