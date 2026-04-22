from gymnasium import Wrapper


class VelocityScaleWrapper(Wrapper):
    """Scale actions from [-1, 1] to [-max_velocity, max_velocity] m/s."""

    def __init__(self, env, max_velocity: float):
        super().__init__(env)
        self.max_velocity = float(max_velocity)

    def step(self, actions, *args, **kwargs):
        return self.env.step(actions * self.max_velocity, *args, **kwargs)
