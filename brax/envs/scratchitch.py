from typing import Tuple

from brax import base
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


class ScratchItch(PipelineEnv):
    
    # TODO: Add docstring
    """
    Add docstring
    """

    # pyformat: enable

    def __init__(
        self,
        ctrl_cost_weight: float = 1e-6,
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
        self.path = epath.resource_path("brax") / "envs/assets/wheelchair_scene.xml"

        mjmodel = mujoco.MjModel.from_xml_path(str(self.path))
        self.sys = mjcf.load_model(mjmodel)
        if backend == "mjx":
            self.sys = self.sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        self.panda_actuators_ids = []
        self.humanoid_actuators_ids = []

        ACTUATOR_IDX = 19
        for i in range(mjmodel.nu):
            actuator_name = mj_id2name(mjmodel, ACTUATOR_IDX, i)
            if actuator_name.startswith("actuator"):
                self.panda_actuators_ids.append(i)
            else:
                self.humanoid_actuators_ids.append(i)

        GEOM_IDX = 5 # 5 is the index of the geom tag in the xml file
        self.target_idx = mj_name2id(mjmodel, GEOM_IDX, "target")

        BODY_IDX = 1
        self.panda_right_finger_idx = mj_name2id(mjmodel, GEOM_IDX, "finger_tip_right")
        self.panda_left_finger_idx = mj_name2id(mjmodel, GEOM_IDX, "finger_tip_left")
        self.panda_effector_idx = mj_name2id(mjmodel, BODY_IDX, "hand")

        self.human_tuarm_idx = mj_name2id(mjmodel, BODY_IDX, "right_upper_arm") # Right human arm tuarm = target arm upper arm
        self.human_tlarm_idx = mj_name2id(mjmodel, BODY_IDX, "right_lower_arm") # Right human arm tlarm = target arm lower arm
        self.contact_force = jax.vmap(contact_force, in_axes=(None, 0, None, None))
        
        n_frames = 4
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=self.sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        # self.actuator_classes = self._get_actuator_classes(self.path)
        # self.humanoid_actuators, self.panda_actuators = self._identify_actuators(self.actuator_classes)

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

        robo_obs = self._get_robo_obs(pipeline_state)
        human_obs = self._get_human_obs(pipeline_state)
        #obs = jp.concatenate((robo_obs, human_obs))
        obs = jp.concatenate((
            robo_obs["position"],
            robo_obs["velocity"],
            robo_obs["distance_to_target"].reshape((1,)),
            robo_obs["tool_orientation"],
            robo_obs["target_pos"],
            robo_obs["human_uarm_pos"],
            robo_obs["human_larm_pos"],
            robo_obs["force_on_target"].reshape((6,)),
            human_obs["position"],
            human_obs["velocity"],
            human_obs["distance_to_target"].reshape((1,)),
            human_obs["tool_orientation"],
            human_obs["target_pos"],
            human_obs["human_uarm_pos"],
            human_obs["human_larm_pos"],
            robo_obs["force_on_target"].reshape((6,)),            
        ))
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_dist": zero,
            "reward_ctrl": zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        ctrl_cost = -self._ctrl_cost_weight * jp.sum(jp.square(action))
        robo_obs = self._get_robo_obs(pipeline_state)
        human_obs = self._get_human_obs(pipeline_state)
        obs = jp.concatenate((
            robo_obs["position"],
            robo_obs["velocity"],
            robo_obs["distance_to_target"].reshape((1,)),
            robo_obs["tool_orientation"],
            robo_obs["target_pos"],
            robo_obs["human_uarm_pos"],
            robo_obs["human_larm_pos"],
            robo_obs["force_on_target"].reshape((6,)),
            human_obs["position"],
            human_obs["velocity"],
            human_obs["distance_to_target"].reshape((1,)),
            human_obs["tool_orientation"],
            human_obs["target_pos"],
            human_obs["human_uarm_pos"],
            human_obs["human_larm_pos"],
            human_obs["force_on_target"].reshape((6,)),
        ))
        r_dist = -robo_obs["distance_to_target"]
        #jax.debug.print("r_dist: {}", r_dist)
        # jax.debug.print("r_dist: {}", r_dist)
        # jax.debug.print("ctrl_cost: {}", ctrl_cost)
        reward = r_dist + ctrl_cost
        done = 0.0
        state.metrics.update(
            reward_dist = r_dist,
            reward_ctrl = ctrl_cost,
        )

        return state.replace(
            pipeline_state=pipeline_state, obs = obs, reward=reward, done=done
        )
        # return (robo_obs, human_obs, target_contact_force)

    # TODO: actually whether or not they make contact should not be used in the observation funciton but in the reward
    # observation should only include the distance from end effectors to the target which I can find with .site_xpos or .geom_xpos
    def _get_robo_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment observations."""
        position = pipeline_state.q
        velocity = pipeline_state.qd
        distance_to_target = self._check_distance_and_contact(pipeline_state, self.target_idx, self.panda_left_finger_idx)
        tool_orientation = pipeline_state.xquat[self.panda_effector_idx]
        target_pos = self._get_geom_and_size(pipeline_state, self.target_idx)
        human_uarm_pos = pipeline_state.xpos[self.human_tuarm_idx]
        human_larm_pos = pipeline_state.xpos[self.human_tlarm_idx]
        forces = self._get_contact_force(pipeline_state)
        force_on_target = self._get_force_on_target(pipeline_state, forces, self.target_idx, self.panda_left_finger_idx, self.panda_right_finger_idx)
        return {
            "position": position,
            "velocity": velocity,
            "distance_to_target": distance_to_target,
            "tool_orientation": tool_orientation,
            "target_pos": target_pos,
            "human_uarm_pos": human_uarm_pos,
            "human_larm_pos": human_larm_pos,
            "forces": forces,
            "force_on_target": force_on_target
        }
        #return jp.concatenate((position, velocity, distance_to_target, tool_orientation, target_pos, human_uarm_pos, human_larm_pos))
    

    # TODO: Forces this is the only way human and robo obs are different
    def _get_human_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment observations."""
        position = pipeline_state.q
        velocity = pipeline_state.qd
        distance_to_target = self._check_distance_and_contact(pipeline_state, self.target_idx, self.panda_left_finger_idx)
        tool_orientation = pipeline_state.xquat[self.panda_effector_idx]
        target_pos = self._get_geom_and_size(pipeline_state, self.target_idx)
        human_uarm_pos = pipeline_state.xpos[self.human_tuarm_idx]
        human_larm_pos = pipeline_state.xpos[self.human_tlarm_idx]
        forces = self._get_contact_force(pipeline_state)
        force_on_target = self._get_force_on_target(pipeline_state, forces, self.target_idx, self.panda_left_finger_idx, self.panda_right_finger_idx)
        return {
            "position": position,
            "velocity": velocity,
            "distance_to_target": distance_to_target,
            "tool_orientation": tool_orientation,
            "target_pos": target_pos,
            "human_uarm_pos": human_uarm_pos,
            "human_larm_pos": human_larm_pos,
            "forces": forces,
            "force_on_target": force_on_target
        }
        #return jp.concatenate((position, velocity, distance_to_target, tool_orientation, target_pos, human_uarm_pos, human_larm_pos))
    
    
    # TODO: Integrates the following functions into the step function
    
    # TODO: Replace some of this with mj_geomDistance function which seems to do alot of this already ...
    def _get_geom_and_size(self, pipeline_state: base.State, geom_id: int) -> Tuple[jax.Array, jax.Array]:
        """Returns the geoms and sizes of the environment."""
        # print("state", pipeline_state)
        # print("model", mjmodel)
        geom_xpos = pipeline_state.geom_xpos[geom_id]
        # geom_size = mjmodel.geom_size[geom_id] # Would it be better to store this as a list in init? 
        # geom_type = jp.array(mjmodel.geom_type)[geom_id]
        

        # def sphere_size(size):
        #     return size[0]  # radius
        
        # def capsule_size(size):
        #     return size[0] + size[1]  # radius + half-length
        
        # def ellipsoid_size(size):
        #     return jp.max(size)  # Max semi-axis
        
        # def cylinder_size(size):
        #     return jp.max(jp.array([size[0], size[1]]))  # Max of radius and half-length
        
        # def box_size(size):
        #     return jp.linalg.norm(size)  # Diagonal half-length
        
        # def default_size(size):
        #     return jp.max(size) # default to max for unsupported types
        
        # size_funcs = [
        #     default_size,
        #     default_size,
        #     sphere_size,
        #     capsule_size,
        #     ellipsoid_size,
        #     cylinder_size,
        #     box_size,
        #     default_size
        # ]


        # effective_size = jax.lax.switch(geom_type, size_funcs, geom_size)

        # broadcast effective_size to match the number of environments
        # effective_size = jp.broadcast_to(effective_size, (geom_xpos.shape[0],))

        # return geom_xpos, effective_size

        return geom_xpos
    
    def _check_distance_and_contact(self, pipeline_state: base.State, geom1_id: int, geom2_id: int) -> Tuple[jax.Array, jax.Array]:
        
        # pos1, size1 = self._get_geom_and_size(pipeline_state, mjmodel, geom1_id)
        # pos2, size2 = self._get_geom_and_size(pipeline_state, mjmodel, geom2_id)

        pos1 = self._get_geom_and_size(pipeline_state, geom1_id)
        pos2 = self._get_geom_and_size(pipeline_state, geom2_id)
    
        center_distance = jp.linalg.norm(pos1 - pos2, axis=-1)
        # surface_distance = center_distance - (size1 + size2)
        # is_contact = surface_distance <= 0
    
        # TODO: The surface distance calculations seem wrong
        # we can jsut use the center distance for now.
        #return (is_contact, surface_distance)
        # return (is_contact, center_distance)
        return center_distance
    
    def _body_geom_ids(self, mjmodel: mujoco.MjModel, body_id: int) -> jax.Array:
        """Returns the geom ids of a body."""
        return jp.where(mjmodel.geom_bodyid == body_id)[0]
    
    def _get_contact_force(self, pipeline_state: base.State) -> jax.Array:
        """Returns the contact force of a geom."""
        forces = [contact_force(self.sys, pipeline_state, i, False) for i in range(pipeline_state.ncon)]

        return jp.array(forces)
    
    def _get_force_on_target(self, pipeline_state: base.State, forces: jax.Array, target_idx: int, panda_left_finger_idx: int, panda_right_finger_idx: int) -> jax.Array:
        """Returns the force on the target."""
        
        target_contacts_left = (pipeline_state.contact.geom == jp.array([panda_left_finger_idx, target_idx])) | (pipeline_state.contact.geom == jp.array([target_idx, panda_left_finger_idx]))
        target_contacts_right = (pipeline_state.contact.geom == jp.array([panda_right_finger_idx, target_idx])) | (pipeline_state.contact.geom == jp.array([target_idx, panda_right_finger_idx]))

        # print(f"target_contacts_left: {target_contacts_left.shape}")
        # print(f"target_contacts_right: {target_contacts_right.shape}")
        # print(f"forces: {forces.shape}")

        mask_left = jp.all(target_contacts_left, axis=1)
        mask_right = jp.all(target_contacts_right, axis=1)

        # print(f"mask_left: {mask_left.shape}")
        # print(f"mask_right: {mask_right}")

        left_force = jp.where(mask_left[:, None], forces, 0)
        right_force = jp.where(mask_right[:, None], forces, 0)
        

        combined_force = jp.concatenate([jp.expand_dims(left_force, axis=0), 
                            jp.expand_dims(right_force, axis=0)], 
                            axis=1)
        # total_force_on_target = jp.sum(, axis=1)
        total_force_on_target = jp.sum(combined_force, axis=1)
        return total_force_on_target
    
    # def _get_force_on_target(self, pipeline_state: base.State, forces: jax.Array, target_idx, panda_left_finger_idx, panda_right_finger_idx) -> jax.Array:
    #     """Returns the force on the target."""
    #     target_contacts_left = pipeline_state.contact.geom == jp.array([target_idx, panda_left_finger_idx])
    #     target_contacts_right = pipeline_state.contact.geom == jp.array([target_idx, self.panda_right_finger_idx])

    #     jax.debug.print("target_contacts_left: {}", jp.all(target_contacts_left))   
    #     target_contact_left_ids  = jp.where(jp.all(target_contacts_left, axis=1))[0]
    #     target_contact_right_ids = jp.where(jp.all(target_contacts_right, axis=1))[0]

    #     all_ids = jp.concatenate((target_contact_left_ids, target_contact_right_ids))

    #     total_force_on_target = jp.sum(forces[all_ids], axis=0)

        
    #     return total_force_on_target


    
    # def _get_target_contacts(self, pipeline_state: base.State, mjmodel: mujoco.MjModel) -> jax.Array:
    #     """Returns the contacts with the target."""
    #     contacts = pipeline_state.contacts
    #     site_id = mj_name2id(mjmodel, 6, "target")
    #     right_finger_id = mj_name2id(mjmodel, 1, "right_finger")
    #     left_finger_id = mj_name2id(mjmodel, 1, "left_finger")  
        
    #     site_matches = contacts.geom1 == site_id
    #     finger_matches = (contacts.geom1 == right_finger_id) | (contacts.geom1 == left_finger_id)
        
    #     matching_contacts = jp.logical_or(
    #     jp.logical_and(site_matches, finger_matches),
    #     jp.logical_and(finger_matches, site_matches)     
    #     )
        
    #     return jp.any(matching_contacts)
    
    # Implemented for step function 

    # def _get_actuator_classes(self, path: str):
    #     """Returns the actuators of the environment."""
    #     tree = ET.parse(path)
    #     root = tree.getroot()
    #     actuators = root.findall(".//actuator")
       
    #     actuator_classes = []
    #     for actuator in actuators:
    #         actuator_class = actuator.get('class', 'default')
    #         actuator_classes.append(actuator_class)

    #     return actuator_classes
    
    # def _identify_actuators(self, actuator_classes):

    #     humanoid_actuators = []
    #     panda_actuators = []

    #     for i, actuator_class in enumerate(actuator_classes):
    #         if actuator_class == 'humanoid':
    #             humanoid_actuators.append(i)
    #         elif actuator_class == 'panda' or actuator_class == 'finger':
    #             panda_actuators.append(i)
        
    #     return humanoid_actuators, panda_actuators
