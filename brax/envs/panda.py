from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
import mujoco
from jax import numpy as jp


class Panda(PipelineEnv):
    """
    ### Description
    Simple test implementation of the panda robotic arm, with the goal of
    moving the end effector with maximum possible downward vertical speed.

    ### Action Space

    The agent take a 6-element vector for actions. The action space is a
    continuous tuple with all elements in `[-1, 1]`, where `action` represents
    the numerical torques applied between *links*

    | Num | Action                   | Control Min | Control Max | Name (in corresponding config) | Joint | Unit         |
    |-----|--------------------------|-------------|-------------|--------------------------------|-------|--------------|
    | 0   | Torque applied on joint1 | -1          | 1           | joint1                         | hinge | torque (N m) |
    | 1   | Torque applied on joint2 | -1          | 1           | joint2                         | hinge | torque (N m) |
    | 2   | Torque applied on joint3 | -1          | 1           | joint3                         | hinge | torque (N m) |
    | 3   | Torque applied on joint4 | -1          | 1           | joint4                         | hinge | torque (N m) |
    | 4   | Torque applied on joint5 | -1          | 1           | joint5                         | hinge | torque (N m) |
    | 5   | Torque applied on joint6 | -1          | 1           | joint6                         | hinge | torque (N m) |

    ### Observation Space

    The state space consists of positional values of different body parts of the
    hopper, followed by the velocities of those individual parts (their
    derivatives) with all the positions ordered before all the velocities.

    The observation is a `ndarray` with shape `(12,)` where the elements
    correspond to the following:

    | Num | Observation                         | Min  | Max | Name (in corresponding config) | Joint | Unit                     |
    |-----|-------------------------------------|------|-----|--------------------------------|-------|--------------------------|
    | 0   | angle of the top                    | -Inf | Inf | joint1                         | hinge | angle (rad)              |
    | 1   | angle of the thigh joint            | -Inf | Inf | joint2                         | hinge | angle (rad)              |
    | 2   | angle of the leg joint              | -Inf | Inf | joint3                         | hinge | angle (rad)              |
    | 3   | angle of the leg joint              | -Inf | Inf | joint4                         | hinge | angle (rad)              |
    | 4   | angle of the leg joint              | -Inf | Inf | joint5                         | hinge | angle (rad)              |
    | 5   | angle of the foot joint             | -Inf | Inf | joint6                         | hinge | angle (rad)              |
    | 6   | angular velocity of the top         | -Inf | Inf | joint1                         | hinge | angular velocity (rad/s) |
    | 7   | angular velocity of the thigh joint | -Inf | Inf | joint2                         | hinge | angular velocity (rad/s) |
    | 8   | angular velocity of the leg joint   | -Inf | Inf | joint3                         | hinge | angular velocity (rad/s) |
    | 9   | angular velocity of the leg joint   | -Inf | Inf | joint4                         | hinge | angular velocity (rad/s) |
    | 10  | angular velocity of the leg joint   | -Inf | Inf | joint5                         | hinge | angular velocity (rad/s) |
    | 11  | angular velocity of the foot joint  | -Inf | Inf | joint6                         | hinge | angular velocity (rad/s) |

    ### Rewards

    The reward consists of three parts:

    - *reward_vertical*: A reward for moving the end effector vertically
    - *reward_ctrl*: A negative reward for penalising the hopper if it takes
      actions that are too large. It is measured as *-coefficient **x**
      sum(action<sup>2</sup>)* where *coefficient* is a parameter set for the
      control and has a default value of 0.001

    ### Starting State

    TODO

    ### Episode Termination

    The episode terminates when any of the following happens:

    1. The episode duration reaches a 1000 timesteps

    """

    # pyformat: enable

    def __init__(
        self,
        vertical_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-6,
        reset_noise_scale=5e-3,
        backend="mjx",
        **kwargs
    ):
        """Creates a Hopper environment.

        Args:
          forward_reward_weight: Weight for the forward reward, i.e. velocity in
            x-direction.
          ctrl_cost_weight: Weight for the control cost.
          reset_noise_scale: Scale of noise to add to reset states.
          backend: str, the physics backend to use
          **kwargs: Arguments that are passed to the base class.
        """
        path = epath.resource_path("brax") / "envs/assets/panda.xml"
        sys = mjcf.load(path)

        n_frames = 4
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._vertical_reward_weight = vertical_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng_pos, rng_vel = jax.random.split(rng, 2)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng_pos, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng_vel, (self.sys.qd_size(),), minval=low, maxval=hi)

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_vertical": zero,
            "reward_ctrl": zero,
            "z_position": zero,
            "z_velocity": zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        END_EFFECTOR_IDX = 6
        z_velocity = (
            pipeline_state.x.pos[END_EFFECTOR_IDX, 2]
            - pipeline_state0.x.pos[END_EFFECTOR_IDX, 2]
        ) / self.dt
        vertical_reward = self._vertical_reward_weight * (-z_velocity)
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        obs = self._get_obs(pipeline_state)
        reward = vertical_reward - ctrl_cost
        done = 0.0
        state.metrics.update(
            reward_vertical=vertical_reward,
            reward_ctrl=-ctrl_cost,
            z_position=pipeline_state.x.pos[END_EFFECTOR_IDX, 2],
            z_velocity=z_velocity,
        )

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment observations."""
        position = pipeline_state.q
        velocity = pipeline_state.qd
        return jp.concatenate((position, velocity))
