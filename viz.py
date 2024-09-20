
import brax
from brax.envs import create
from brax.io import html
import jax

# Create the environment (e.g., "ant")
env = create(env_name="panda")

# Initialize a random number generator key
rng = jax.random.PRNGKey(seed=0)

# Reset the environment with the rng
state = env.reset(rng=rng)

# Render the environment to HTML
html_str = html.render(env.sys, [state.pipeline_state])
with open('brax_env.html', 'w') as f:
    f.write(html_str)
