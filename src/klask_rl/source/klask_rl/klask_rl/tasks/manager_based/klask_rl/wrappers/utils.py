def find_wrapper(env, wrapper_type):
    """Recursively searches for a wrapper of a given type."""
    while not isinstance(env, wrapper_type):
        env = env.env  # Move to the next layer
    if isinstance(env, wrapper_type):
        return env  # Found the wrapper
    return None  # Wrapper not found


def configure_domain_randomization(env_cfg, dr_cfg):
    """Configure domain randomization events on *env_cfg* from a config dict.

    Call this BEFORE ``gym.make()`` so that Isaac Lab's event manager either
    receives the correct ranges or never registers the placeholder events.

    Parameters
    ----------
    env_cfg : object
        The Isaac Lab environment config (must have ``env_cfg.events``).
    dr_cfg : dict | None
        The ``domain_randomization`` section from the training YAML, already
        resolved to a plain dict.  ``None`` disables all DR events.
    """
    event_names = ["add_ball_mass", "randomize_material_ball",
                   "randomize_material_klask", "randomize_actuator"]

    if dr_cfg is None:
        for name in event_names:
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)
        return

    # Ball mass
    bm = dr_cfg.get("ball_mass", {})
    if bm.get("enable", False):
        env_cfg.events.add_ball_mass.params["mass_distribution_params"] = tuple(bm["range"])
    else:
        env_cfg.events.add_ball_mass = None

    # Material: ball
    mb = dr_cfg.get("material_ball", {})
    if mb.get("enable", False):
        env_cfg.events.randomize_material_ball.params["static_friction_range"] = tuple(mb["static_friction_range"])
        env_cfg.events.randomize_material_ball.params["dynamic_friction_range"] = tuple(mb["dynamic_friction_range"])
        env_cfg.events.randomize_material_ball.params["restitution_range"] = tuple(mb["restitution_range"])
    else:
        env_cfg.events.randomize_material_ball = None

    # Material: klask
    mk = dr_cfg.get("material_klask", {})
    if mk.get("enable", False):
        env_cfg.events.randomize_material_klask.params["static_friction_range"] = tuple(mk["static_friction_range"])
        env_cfg.events.randomize_material_klask.params["dynamic_friction_range"] = tuple(mk["dynamic_friction_range"])
        env_cfg.events.randomize_material_klask.params["restitution_range"] = tuple(mk["restitution_range"])
    else:
        env_cfg.events.randomize_material_klask = None

    # Actuator gains
    ac = dr_cfg.get("actuator", {})
    if ac.get("enable", False):
        env_cfg.events.randomize_actuator.params["stiffness_distribution_params"] = tuple(ac["stiffness_range"])
        env_cfg.events.randomize_actuator.params["damping_distribution_params"] = tuple(ac["damping_range"])
    else:
        env_cfg.events.randomize_actuator = None
