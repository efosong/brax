

## Installation
```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -e .
pip install -U "jax[cuda12]"
```

## Running

Test installation
```
python brax/envs/env_test.py
```
To visualise an xml file:
```
python viz_xml.py
```


To visualise an environment
```
python viz.py
```


## menagerie

```
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```