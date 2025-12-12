from dm_control.rl.control import Environment

class PhysicsEnv(Environment):
    def __init__(self, physics, timestep=0.02):
        self._physics = physics
        self._timestep = timestep
        self._step_count = 0

    def reset(self):
        pass

    def step(self, action):
        pass

    def action_spec(self):
        # no actions → return empty spec
        from dm_control import specs
        return specs.BoundedArray(shape=(0,), dtype=float, minimum=0, maximum=0)

    @property
    def physics(self):
        return self._physics
