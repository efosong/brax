from typing import Tuple

from brax import base
from brax import math as bmath
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
import mujoco 
from mujoco import mj_id2name, mj_name2id
from enum import IntEnum
from mujoco.mjx._src.support import contact_force
import numpy as np
from mujoco import mjtObj

# import xml.etree.ElementTree as ET

class GeomType(IntEnum):
    PLANE = 0
    HFIELD = 1
    SPHERE = 2
    CAPSULE = 3
    ELLIPSOID = 4
    CYLINDER = 5
    BOX = 6
    MESH = 7


class Stand(PipelineEnv):
    
    # TODO: Add docstring
    """
    Add docstring
    """

    # pyformat: enable

    def __init__(
        self,
        ctrl_cost_weight: float = 1e-6,
        dist_reward_weight: float = 1.0,
        dist_scale: float = 0.1,
        reset_noise_scale=5e-3,
        backend="mjx",
        **kwargs
    ):
        """Creates a Hopper environment.

        Args:
          ctrl_cost_weight: Weight for the control cost.
          reset_noise_scale: Scale of noise to add to reset states.
          backend: str, the physics backend to use
          **kwargs: Arguments that are passed to the base class.
        """
        self.path = epath.resource_path("brax") / "envs/assets/torobo2_standard.xml"

        mjmodel = mujoco.MjModel.from_xml_path(str(self.path))
        self.sys = mjcf.load_model(mjmodel)
        if backend == "mjx":
            self.sys = self.sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON, # Try mjSOL_CG for better stability
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 4,  # max number of iterations for main solver, O(n)
                    "opt.ls_iterations": 4, # helps with stability, O(iterations x ls_iterations)
                    "opt.timestep": 0.001
                }
            )
    
        # sort actuators - WHERE IS THIS ACTUALLY USED?
        self.robot_actuators_ids = []
        self.humanoid_actuators_ids = []
        ACTUATOR_IDX = 19
        for i in range(mjmodel.nu):
            actuator_name = mj_id2name(mjmodel, ACTUATOR_IDX, i)
            if actuator_name.startswith("robot"):
                self.robot_actuators_ids.append(i)
            else:
                self.humanoid_actuators_ids.append(i)

        GEOM_IDX = 5 # 5 is the index of the geom tag in the xml file
        BODY_IDX = 1

        # we want the human waist position for rewards
        self.human_waist_geom_idx = (mj_name2id(mjmodel, GEOM_IDX, "lwaist"))
        
        # Get this from the mujoco viewer, count manually
        num_head_joints = 3
        num_torso_joints = 3
        num_base_joints = 3
        num_arm_joints = 7
        self.num_robot_joints = num_head_joints + num_torso_joints + 2*num_arm_joints + num_base_joints

        # we add 1 to ensure end index included
        joint_names = [mjmodel.joint(i).name for i in range(mjmodel.njnt)]
        self.human_joint_id_start = joint_names.index('abdomen_z')  # This should return 1
        self.human_joint_id_end = joint_names.index('left_elbow') + 1  # This should return 17
        self.robot_joint_id_start = joint_names.index('base_joint_trans_x')  # This should return 18
        self.robot_joint_id_end = joint_names.index('head/joint_3') + 1 # This should return 24

        
        # Retrieve joint limits
        # self.upper_joint_limits = mjmodel.jnt_range[:, 0]
        # self.lower_joint_limits = mjmodel.jnt_range[:, 1]
        # self.robot_upper_joint_limits = self.upper_joint_limits[self.robot_joint_id_start:self.robot_joint_id_end]
        # self.robot_lower_joint_limits = self.lower_joint_limits[self.robot_joint_id_start:self.robot_joint_id_end]
        # self.human_upper_joint_limits = self.upper_joint_limits[self.human_joint_id_start:self.human_joint_id_end]
        # self.human_lower_joint_limits = self.lower_joint_limits[self.human_joint_id_start:self.human_joint_id_end]
        n_frames = 4
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=self.sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._dist_reward_weight = dist_reward_weight
        self._dist_scale = dist_scale
        self._reset_noise_scale = reset_noise_scale


    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng_pose, rng_scratch = jax.random.split(rng, 2)

        # Add small positional and velocity noise to initialisation
        rng_pos, rng_vel = jax.random.split(rng_pose, 2)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        # TODO: BRING BACK THE INITIAL POS
        # init_q = self.sys.mj_model.keyframe("init").qpos
        # qpos = init_q + jax.random.uniform(
        #     rng_pos, (self.sys.q_size(),), minval=low, maxval=hi
        # )
        qpos = jax.random.uniform(
            rng_pos, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng_vel, (self.sys.qd_size(),), minval=low, maxval=hi)

        pipeline_state = self.pipeline_init(qpos, qvel)

        # NOTE: IF THE LENGTH OF ANY OBSERVATIONS CHANGE, YOU MUST UPDATE THIS HERE:
        # jaxmarl/environments/mabrax/mappings.py#L111
        info = {}
        robo_obs = self._get_robo_obs(pipeline_state, info)
        human_obs = self._get_human_obs(pipeline_state, info)
        obs = jp.concatenate((
            # 23 + 23 + 17 + 3 = 46 + 20 = 66
            robo_obs["robo_joint_angles"],
            robo_obs["robo_joint_vel"],
            robo_obs["human_joint_angles"],  
            robo_obs["human_waist_pos"],
            # human obs length = 17 + 17 + 3 = 37
            human_obs["human_joint_angles"],
            human_obs["human_joint_vel"],       
            human_obs["human_waist_pos"],          
        ))

        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward": zero,
        }
        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, rng: jax.Array, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        # recompute obs
        robo_obs = self._get_robo_obs(pipeline_state, state.info)
        human_obs = self._get_human_obs(pipeline_state, state.info)
        obs = jp.concatenate((
            robo_obs["robo_joint_angles"],
            robo_obs["robo_joint_vel"],
            robo_obs["human_joint_angles"],  
            robo_obs["human_waist_pos"],
            human_obs["human_joint_angles"],  
            human_obs["human_joint_vel"],       
            human_obs["human_waist_pos"],      
        ))

        # robo_obs_length = jp.concatenate((
        #     robo_obs["robo_joint_angles"],
        #     robo_obs["robo_joint_vel"]))
        
        # human_obs_length = jp.concatenate((
        #     human_obs["human_joint_angles"],         
        #     human_obs["human_waist_pos"],    
        # ))
        # self.print_function("robot", robo_obs_length.size)
        # self.print_function("human", human_obs_length.size)

        human_waist_height = human_obs["human_waist_pos"][2]
        standing_reward = human_waist_height

        # sum rewards
        reward = standing_reward

        done = 0.0

        # logging
        state.metrics.update(
            reward = reward,
        )

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done
        )
    
    def print_function(self, name, x):
        jax.debug.print(f"{name}: {x}")


    def _get_robo_obs(self, pipeline_state, state_info) -> jax.Array:
        """Returns the environment observations."""

        # normalise joint angles to range [0, 1], then [-1, 1]
        robo_joint_angles = pipeline_state.qpos[self.robot_joint_id_start:self.robot_joint_id_end]
        robo_joint_vel = pipeline_state.qd[self.robot_joint_id_start:self.robot_joint_id_end]
        # TODO: normalise joint velocities?

        # robot needs to know human pos in order to move
        human_waist_pos = pipeline_state.xpos[self.human_waist_geom_idx]      
        human_joint_angles = pipeline_state.qpos[self.human_joint_id_start:self.human_joint_id_end]

        return {
            "robo_joint_angles": robo_joint_angles,
            "robo_joint_vel": robo_joint_vel,
            "human_joint_angles": human_joint_angles,
            "human_waist_pos": human_waist_pos        
        }
    

    # TODO: Forces this is the only way human and robo obs are different
    def _get_human_obs(self, pipeline_state, state_info) -> jax.Array:
        """Returns the environment observations."""

        human_joint_angles = pipeline_state.qpos[self.human_joint_id_start:self.human_joint_id_end]
        human_joint_vel = pipeline_state.qd[self.human_joint_id_start:self.human_joint_id_end]

        human_waist_pos = pipeline_state.xpos[self.human_waist_geom_idx]      

        return {
            "human_joint_angles": human_joint_angles,
            "human_joint_vel": human_joint_vel,
            "human_waist_pos": human_waist_pos
        }
    
    
