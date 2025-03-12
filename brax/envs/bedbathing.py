from typing import Tuple

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
from jax.scipy.spatial.transform import Rotation
import mujoco 
from mujoco import mj_id2name, mj_name2id
from enum import IntEnum
from mujoco.mjx._src.support import contact_force
import numpy as np
# from brax import idcontacts

def contact_id(pipeline_state: State, id1: int, id2: int) -> int:
    """Returns the contact id between two geom ids."""
    mask = (pipeline_state.contact.geom == jp.array([id1, id2])) | (pipeline_state.contact.geom == jp.array([id2, id1])) 
    id = jp.all(mask, axis=0)   
    return id

class BedBathing(PipelineEnv):
    
    # TODO: Add docstring
    """
    Add docstring
    """

    # pyformat: enable

    def __init__(
        self,
        ctrl_cost_weight: float = 1e-6,
        dist_reward_weight: float = 0.1,
        dist_scale: float = 0.1,
        wiping_reward_weight: float = 1.0,
        reset_noise_scale=5e-3,
        backend="mjx",
        n_targets: int = 0,
        target_threshold: float = 0.1,
        **kwargs
    ):
        """Creates a BedBathing Environment.

        Args:
          ctrl_cost_weight: Weight for the control cost.
          reset_noise_scale: Scale of noise to add to reset states.
          backend: str, the physics backend to use
          **kwargs: Arguments that are passed to the base class.
        """
        self.path = epath.resource_path("brax") / "envs/assets/bed_scene.xml"

        mjmodel = mujoco.MjModel.from_xml_path(str(self.path))
        self.sys = mjcf.load_model(mjmodel)
        if backend == "mjx":
            self.sys = self.sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 1,
                    "opt.timestep": 0.001
                }
            )

        GEOM_IDX = mujoco.mjtObj.mjOBJ_GEOM
        BODY_IDX = mujoco.mjtObj.mjOBJ_BODY
        ACTUATOR_IDX = mujoco.mjtObj.mjOBJ_ACTUATOR
        SITE_IDX = mujoco.mjtObj.mjOBJ_SITE
    
        self.panda_actuators_ids = []
        self.humanoid_actuators_ids = []

        # TODO: Is this particioning of actuators useful in the training pipeline? 
        for i in range(mjmodel.nu):
            actuator_name = mj_id2name(mjmodel, ACTUATOR_IDX, i)
            if actuator_name.startswith("actuator"):
                self.panda_actuators_ids.append(i)
            else:
                self.humanoid_actuators_ids.append(i)

        self.panda_wiper_idx = mj_name2id(mjmodel, GEOM_IDX, "wiper_pad")
        self.panda_wiper_center_idx = mj_name2id(mjmodel, SITE_IDX, "wiper_centre")
        self.panda_wiper_body_idx = mj_name2id(mjmodel, BODY_IDX, "wiper")

        # self.targets is a fixed array containing the int ids of the target sites 
        self.n_targets = 10
        target_idxs = [f"target_{id}" for id in range(self.n_targets)]
        self.targets = jp.array([mj_name2id(mjmodel, SITE_IDX, idx) for idx in target_idxs], dtype=jp.int32)

        self.human_tuarm_idx = mj_name2id(mjmodel, BODY_IDX, "right_upper_arm") # Right human arm tuarm = target arm upper arm
        self.human_tlarm_idx = mj_name2id(mjmodel, BODY_IDX, "right_lower_arm") # Right human arm tlarm = target arm lower arm

        self.human_tuarm_geom = mj_name2id(mjmodel, GEOM_IDX, "right_uarm")
        self.human_tlarm_geom = mj_name2id(mjmodel, GEOM_IDX, "right_larm")

        self.human_uarm_size = mjmodel.geom_size[self.human_tuarm_geom]
        self.human_larm_size = mjmodel.geom_size[self.human_tlarm_geom]
        
        # self.contact_force = jax.vmap(contact_force, in_axes=(None, 0, None, None))
        # self.TARGET_CONTACT_ID = 294
        
        # TODO: Update these indexes once XML is updated or write some sort of helper function to get these
        self.UARM_TOOL_CONTACT_ID = 58
        self.LARM_TOOL_CONTACT_ID = 62
        
        # we add 1 to ensure end index included
        joint_names = [mjmodel.joint(i).name for i in range(mjmodel.njnt)]
        self.human_joint_id_start = joint_names.index('abdomen_z')  # This should return 1
        self.human_joint_id_end = joint_names.index('left_elbow') + 1  # This should return 17
        self.panda_joint_id_start = joint_names.index('joint1')  # This should return 18
        self.panda_joint_id_end = joint_names.index('joint7') + 1 # This should return 24

        # Retrieve joint limits
        self.upper_joint_limits = mjmodel.jnt_range[:, 0]
        self.lower_joint_limits = mjmodel.jnt_range[:, 1]
        self.robot_upper_joint_limits = self.upper_joint_limits[self.panda_joint_id_start:self.panda_joint_id_end]
        self.robot_lower_joint_limits = self.lower_joint_limits[self.panda_joint_id_start:self.panda_joint_id_end]
        self.human_upper_joint_limits = self.upper_joint_limits[self.human_joint_id_start:self.human_joint_id_end]
        self.human_lower_joint_limits = self.lower_joint_limits[self.human_joint_id_start:self.human_joint_id_end]

        n_frames = 4
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=self.sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._dist_reward_weight = dist_reward_weight
        self._wiping_reward_weight = wiping_reward_weight
        self._dist_scale = dist_scale
        self._reset_noise_scale = reset_noise_scale


    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""       

        rng_pos, rng_vel = jax.random.split(rng, 2)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        init_q = self.sys.mj_model.keyframe("init").qpos
        #init_q = self.sys.init_q
        qpos = init_q + jax.random.uniform(
            rng_pos, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng_vel, (self.sys.qd_size(),), minval=low, maxval=hi)

        pipeline_state = self.pipeline_init(qpos, qvel)

        # uarm_contact_id = contact_id(pipeline_state, self.human_tuarm_idx, self.panda_wiper_idx)
        # larm_contact_id = contact_id(pipeline_state, self.human_tlarm_idx, self.panda_wiper_idx)
        # self.LARM_TOOL_CONTACT_ID = jp.int32(larm_contact_id)
        # self.UARM_TOOL_CONTACT_ID = jp.int32(uarm_contact_id)

        target_index = 0
        robo_obs = self._get_robo_obs(pipeline_state, target_index)    
        human_obs = self._get_human_obs(pipeline_state)
        #obs = jp.concatenate((robo_obs, human_obs))
        obs = jp.concatenate((
            # robot obs length = 7 + 7 + 3 + 4 + 6 + 6 + 4 = 37
            robo_obs["robo_joint_angles"],
            robo_obs["robo_joint_vel"],
            robo_obs["tool_position"],
            robo_obs["tool_orientation"],
            robo_obs["human_uarm_pos"],
            robo_obs["human_larm_pos"],
            robo_obs["force_on_tool"].reshape((6,)),
            robo_obs["wiper_target_dist"].reshape((3,)),
            robo_obs["wiper_target_dist_euclidean"].reshape((1,)),  
            # human = 17 + 17 + 7 + 6 + 6 = 53
            human_obs["human_joint_angles"],
            human_obs["human_joint_vel"],   
            human_obs["tool_position"],
            human_obs["tool_orientation"],
            human_obs["human_uarm_pos"],
            human_obs["human_larm_pos"],
            human_obs["force_on_human"].reshape((6,)),

        ))
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_dist": zero,
            "reward_ctrl": zero,
            "reward_wiping": zero,
            "weighted_reward_dist": zero,
            "weighted_reward_ctrl": zero,
            "weighted_reward_wiping": zero,
            "n_contacts": zero,
            "done": zero
        }
        info = {"target_index": target_index}
        return State(pipeline_state, obs, reward, done, metrics, info)
    

    def step(self, rng: jax.Array, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        target_index = state.info["target_index"]
        robo_obs = self._get_robo_obs(pipeline_state, target_index)        
        human_obs = self._get_human_obs(pipeline_state)

        obs = jp.concatenate((
            robo_obs["robo_joint_angles"],
            robo_obs["robo_joint_vel"],
            robo_obs["tool_position"],
            robo_obs["tool_orientation"],
            robo_obs["human_uarm_pos"],
            robo_obs["human_larm_pos"],
            robo_obs["force_on_tool"].reshape((6,)),
            robo_obs["wiper_target_dist"].reshape((3,)),
            robo_obs["wiper_target_dist_euclidean"].reshape((1,)),  
            human_obs["human_joint_angles"],
            human_obs["human_joint_vel"],   
            human_obs["tool_position"],
            human_obs["tool_orientation"],
            human_obs["human_uarm_pos"],
            human_obs["human_larm_pos"],
            human_obs["force_on_human"].reshape((6,)),        
        ))

        dist = robo_obs["wiper_target_dist_euclidean"]
        distance_threshold = dist < 0.1

        contact_forces = robo_obs["force_on_tool"]
        non_zero_forces = jp.any(contact_forces != 0.0)

        # jp.logical_and(below_threshold, non_zero_forces)
        # set contact from 1->0 
        made_contact = jp.where(distance_threshold, 1, 0)
        wiping_reward = made_contact.astype(jp.float32) # Specify reward as float32
        move_index = made_contact.astype(np.int32)  # Specify index increment as int32 

        # Update target index in the state
        new_target_index = state.info["target_index"] + move_index

        # Ensure we don't exceed n_targets
        new_target_index = jp.minimum(new_target_index, self.n_targets)
        info = {"target_index": new_target_index}

        # Set done when we've contacted all targets
        fake_done = (new_target_index >= self.n_targets).astype(jp.float32)
        done = 0.0
        
        # distance reward to current target
        self._dist_scale = 1
        r_dist = (1 - jp.tanh(dist / self._dist_scale))

        # TODO: Add human preference rewards
        self._wiping_reward_weight = 1

        # penalise joint velocities
        ctrl_cost = -jp.sum(jp.square(robo_obs["robo_joint_vel"]))

        reward = self._dist_reward_weight*r_dist + self._ctrl_cost_weight*ctrl_cost + self._wiping_reward_weight*wiping_reward
                
        # also in resset
        state.metrics.update(
            reward_dist = r_dist,
            reward_ctrl = ctrl_cost,
            reward_wiping = wiping_reward,
            weighted_reward_dist = self._dist_reward_weight*r_dist,
            weighted_reward_ctrl = self._ctrl_cost_weight*ctrl_cost,
            weighted_reward_wiping = self._wiping_reward_weight*wiping_reward,
            n_contacts = new_target_index.astype(jp.float32),
            done = fake_done
        )

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            info=state.info | info,
        )

    def _get_robo_obs(self, pipeline_state: base.State, target_index) -> jax.Array:
        """Returns the environment observations."""
        # proprioception
        robo_joint_angles = pipeline_state.qpos[self.panda_joint_id_start:self.panda_joint_id_end]
        normalised_robo_joint_angles = (robo_joint_angles - self.robot_lower_joint_limits) / (self.robot_upper_joint_limits - self.robot_lower_joint_limits)
        normalised_robo_joint_angles = 2 * normalised_robo_joint_angles - 1
        robo_joint_vel = pipeline_state.qd[self.panda_joint_id_start:self.panda_joint_id_end]
        tool_position = pipeline_state.site_xpos[self.panda_wiper_center_idx]
        tool_orientation = pipeline_state.xquat[self.panda_wiper_body_idx]

        # tactile
        force_on_tool = self._get_force_on_tool(pipeline_state, self.UARM_TOOL_CONTACT_ID, self.LARM_TOOL_CONTACT_ID)

        human_uarm_pos = pipeline_state.xpos[self.human_tuarm_idx]
        human_larm_pos = pipeline_state.xpos[self.human_tlarm_idx]

        # mask distances and get closest target 
        next_target_idx = self.targets.at[target_index].get()

        next_target = pipeline_state.site_xpos[next_target_idx]
        wiper_target_dist = next_target - tool_position
        wiper_target_dist_euclidean = jp.linalg.norm(wiper_target_dist)

        return {
            # proprioception
            "robo_joint_angles": normalised_robo_joint_angles,
            "robo_joint_vel": robo_joint_vel,
            "tool_position": tool_position,
            "tool_orientation": tool_orientation,
            # tactile
            "force_on_tool": force_on_tool, 
            # gt
            "human_uarm_pos": human_uarm_pos,
            "human_larm_pos": human_larm_pos,
            "wiper_target_dist": wiper_target_dist,
            "wiper_target_dist_euclidean": wiper_target_dist_euclidean  
        }
       
    def _get_human_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment observations"""
        # proprioception
        human_joint_angles = pipeline_state.qpos[self.human_joint_id_start:self.human_joint_id_end]
        normalised_human_joint_angles = (human_joint_angles - self.human_lower_joint_limits) / (self.human_upper_joint_limits - self.human_lower_joint_limits)
        normalised_human_joint_angles = 2 * normalised_human_joint_angles - 1
        human_joint_vel = pipeline_state.qd[self.human_joint_id_start:self.human_joint_id_end]
        human_uarm_pos = pipeline_state.xpos[self.human_tuarm_idx]
        human_larm_pos = pipeline_state.xpos[self.human_tlarm_idx]

        force_on_human = self._get_force_on_tool(pipeline_state, self.UARM_TOOL_CONTACT_ID, self.LARM_TOOL_CONTACT_ID)
        tool_position = pipeline_state.site_xpos[self.panda_wiper_center_idx]
        tool_orientation = pipeline_state.xquat[self.panda_wiper_body_idx]

        return {
            "human_joint_angles": normalised_human_joint_angles,
            "tool_position": tool_position,
            "tool_orientation": tool_orientation,
            "human_joint_vel": human_joint_vel,
            "human_uarm_pos": human_uarm_pos,
            "human_larm_pos": human_larm_pos,
            "force_on_human": force_on_human,
        } 
    
    def _get_force_on_tool(self, pipeline_state, uarm_tool_id: int, larm_id:int) -> jax.Array:
        """Return the force on the tool"""
        tool_uarm = contact_force(self.sys, pipeline_state, uarm_tool_id, False)
        tool_larm = contact_force(self.sys, pipeline_state, larm_id, False)

        return jp.sum(jp.vstack((tool_uarm, tool_larm)), axis=0)
    
  