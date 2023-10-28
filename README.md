# Representation Erasure Option Learning
This repo contains code accompaning the paper, Enhancing Sample Efficiency in Option Learning with Representation Erasure(AISTATS 2024). It includes code for Representation Erasure Option Learning (REOL) to run all the experiments described in the paper.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Contents:
- [Control Experiments](#control-experiments-tmaze--halfcheetah)
- [Visual Navigation Experiments](#visual-navigation-experiments-miniworld)
- [Additional Material](#additional-material)
- [Citations](#citations)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Control Experiments (TMaze & HalfCheetah)


#### Dependencies
To install dependencies for control experiments: run the following commands:
```
conda create -n intfc python=3.6
conda actvate intfc
pip install tensorflow
pip install -e . (in the main directory)
pip install gym==0.9.3
pip install mujoco-py==0.5.1
brew install mpich
pip install mpi4py
```


#### Usage
To run the code with TMaze experiments, use:
```python run_mujoco_reol.py --env TMaze  --opt 2 --seed 2 --switch```

To run the code with HalfCheetah experiments, use:
```python run_mujoco_reol.py --env HalfCheetahDir-v1  --opt 2 --seed 2 --switch```

To run the baseline MOC, use:

```python run_mujoco_moc.py --env TMaze  --opt 2 --seed 2 --switch```

```python run_mujoco_moc.py --env HalfCheetahDir-v1  --opt 2 --seed 2 --switch```

To run the baseline IOC use:

```python run_mujoco.py --env TMaze  --opt 2 --seed 2 --switch```

```python run_mujoco.py --env HalfCheetahDir-v1  --opt 2 --seed 2 --switch```

To run the baseline option-critic, use the flag `--nointfc` in the above script:

```python run_mujoco.py --env TMaze  --opt 2 --seed 2 --nointfc --switch```

```python run_mujoco.py --env HalfCheetahDir-v1  --opt 2 --seed 2 --nointfc --switch```


#### Running experiments on slurm
To run the code on compute canada or any slurm cluster, make sure you have installed all dependencies and created a conda environment _intf_. 
Now, use the script launcher_mujoco.sh wherein you would need to add account and add username and then run:

```
chmod +x launcher_mujoco.sh
./launcher_mujoco.sh
```

To run the baseline option-critic, use the flag `--nointfc` in the above script:
```
k="xvfb-run -n "${port[$count]}" -s \"-screen 0 1024x768x24 -ac +extension GLX +render -noreset\" python run_mujoco_reol.py --env "$envname" --saves --opt 2 --seed ${_seed} --mainlr ${_mainlr} --piolr ${_piolr} --switch --nointfc --wsaves"
```

#### Performance and Visualizations
To load and run a trained agent, use:
```
python run_mujoco_reol.py --env HalfCheetahDir-v1 --epoch 400 --seed 0
```
where _epoch_ would be the training epoch at which you want to visualize the learned agent. This assumes that the saved model directory is in the ppoc_int folder.



------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Visual Navigation Experiments (Miniworld)

#### Dependencies
To install dependencies for miniworld experiments: run the following commands:
```
conda create -n intfc python=3.6
conda actvate intfc
pip install tensorflow
pip install -e . (in first directory of baselines)
brew install mpich
pip install mpi4py
pip install matplotlib
# to run the code with miniworld
pip install gym==0.10.5
```

To install [miniworld](https://github.com/maximecb/gym-miniworld): follow these [installation instructions](https://github.com/maximecb/gym-miniworld#installation).

Since the cnn policy code is much slower than mujoco experiments, the optimal way to run is using a cluster. To run miniworld headless and training on a cluster, follow these instructions [here](https://github.com/maximecb/gym-miniworld/blob/master/docs/troubleshooting.md#running-headless-and-training-on-aws).


#### Usage
To run the code headless for oneroom task with transfer, use:
```
xvfb-run -n 4005 -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python run_miniw.py --env MiniWorld-OneRoom-v0 --seed 5 --opt 2 --saves --mainlr 1e-4 --intlr 9e-5 --switch --wsaves
```


#### Running experiments on slurm
To run the code on compute canada or any slurm cluster, make sure you have installed all dependencies and created a conda environment _intf_. 
Now, use the script launcher_miniworld.sh wherein you would need to add account and add username and then run:
```
chmod +x launcher_miniworld.sh
./launcher_miniworld.sh
```

Please note that to ensure that miniworld code runs correctly headless, we here make sure we specify an exclusive port per run. 
If the port# overlaps for multiple jobs, the jobs will fail. Ideally there has to be a better way to do this, but this is the one we found easiest to make it work. Depending on how many jobs you want to launch (e.x. runs/seeds), set the range for port accordingly.


To run the baseline option-critic, use the flag `--nointfc` in the above script in the run command.



#### Performance and Visualizations
To plot the learning curves, use the script: `miniworld/baselines/ppoc_int/plot_res.py` with appropiate settings. 

To visualize the trajectories of trained agents: make the following changes in your local installation of the miniworld environment code: https://github.com/kkhetarpal/gym-miniworld/commits/master
Load and run the trained agent to visualize the trajectory of the trained agents with a 2-D top-view of the 3D oneroom.

To load and run a trained agent, use:
```
python run_miniw.py --env MiniWorld-OneRoom-v0 --epoch 480 --seed 0
```
where _epoch_ would be the training epoch at which you want to visualize the learned agent. This assumes that the saved model directory is in the ppoc_int folder.




## Additional Material

* Poster presented at NeurIPS 2019, Deep RL Workshop, Learning Transferable Skills Workshop can be found ([here](https://kkhetarpal.files.wordpress.com/2019/12/neurips_drl_optionsofinterest_poster.pdf)).
* Preliminary ideas presented in AAAI 2019, Student Abstract track, Selected as a finalist in 3MT Thesis Competition ([paper link](https://www.aaai.org/ojs/index.php/AAAI/article/view/5114)), ([poster link](https://kkhetarpal.files.wordpress.com/2019/08/poster_interestfunctions.pdf)).


## Citations
* The fourrooms experiment is built on the Option-Critic, [2017](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14858/14328) [tabular code](https://github.com/jeanharb/option_critic/tree/master/fourrooms).
* The PPOC, [2017](https://arxiv.org/pdf/1712.00004.pdf) baselines [code](https://github.com/mklissa/PPOC) serves as base to our function approximation experiments.
* To install Mujoco, please visit their [website](https://www.roboti.us/license.html) and acquire a free student license.
* For any issues you face with setting up miniworld, please visit their [troubleshooting](https://github.com/maximecb/gym-miniworld/blob/master/docs/troubleshooting.md) page.

