import mujoco
import glfw
import time
import os

# Path to your XML file
xml_file_path = "/home/elle/code/external/brax/brax/envs/assets/shadow.xml"  # Replace this with the path to your MJX file

# Load the MuJoCo model from the XML file
model = mujoco.MjModel.from_xml_path(xml_file_path)

# Create a data object to hold the simulation state
data = mujoco.MjData(model)

# Initialize GLFW to create a window
if not glfw.init():
    raise Exception("Could not initialize GLFW")

# Create a window using GLFW
window = glfw.create_window(1280, 720, "MuJoCo Viewer", None, None)
glfw.make_context_current(window)

# Create a scene object to hold the visual elements
scene = mujoco.MjvScene(model, maxgeom=1000)

# Create a camera object
cam = mujoco.MjvCamera()

# Set camera parameters (you can adjust these values)
cam.lookat[:] = [0.0, 0.0, 0.0]   # The point the camera looks at (x, y, z)
cam.distance = 1.0                 # Distance from the camera to the lookat point
cam.azimuth = 270                   # Horizontal angle in degrees (0 is along x-axis)
cam.elevation = -20                # Vertical angle in degrees (0 is horizontal)

# # Set up default camera position
# mujoco.mjv_defaultCamera(cam)

# Set up a context for rendering
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# Simulation loop
while not glfw.window_should_close(window):
    # Step the simulation (optional for static visualization)
    mujoco.mj_step(model, data)

    # Get the width and height of the window
    width, height = glfw.get_framebuffer_size(window)

    # Create a viewport
    viewport = mujoco.MjrRect(0, 0, width, height)

    # Update the scene with the latest simulation data
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), mujoco.MjvPerturb(), cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

    # Render the scene in the current window
    mujoco.mjr_render(viewport, scene, context)

    # Swap buffers to display the rendered image
    glfw.swap_buffers(window)

    # Poll for input events
    glfw.poll_events()

    # Slow down the loop for visibility
    time.sleep(1 / 60)

# Clean up GLFW when done
glfw.terminate()