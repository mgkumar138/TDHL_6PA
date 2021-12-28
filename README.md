# TDHL_6PA
Training classical agents and agents with a single hidden layer to learn 6 paired associations in a navigationt task with sparse rewards
Agents are trained using Temporal Difference Error modulated Hebbian Plascticity

Run 6pa_classic.py or 6pa_hidden.py to start the 6PA navigation task with the relevant agents.

Other scripts include:
  - 1pa_{} trains the agent to learn single reward locations
  - 6pa_{} trains the agent to learn 6PAs with cues presented throughout the trial
  - wkm_{} trains the agent with a bump attractor to learn 6PAs with transient cues presented
  - 16pa_{} trains the agent with hidden layer to learn 16 PAs with different hyperparameter conditions
  - 6pa_hidden_a2c.py is the agent with a single nonlinear hidden layer and discrete actions trained by Advantage Actor Critic (A2C)


# A Nonlinear Hidden Layer Enables Actor–Critic Agents to Learn Multiple Paired Association Navigation

This repository contains the  
- Code to obtain the results described in the paper https://arxiv.org/abs/2106.13541
      
The main result of the paper is to demonstrate the gradual learning and navigation to single and multiple cued locations
 using an Actor-Critic agent trained by temporal difference error modulated Hebbian plasticity.


9 agents were evaluated in four tasks and script begins with the task type with the follwing nomenclature:
- Single targets - 1pa*
- Six cue-targets - 6pa*    
- 16 cue-targets - 16pa*
- Siz cue-targets with cue given only at the start of the trial - wkm*

and ends with the following nomenclature:
- Classical Actor Critic with no plasticity (No Plasticity) - *control
- Classical Actor Critic (Classic) - *classic
- Classical Actor Critic with multiples of same input (Expanded Classic) - *expclass
- Agent with a linear feedforward layer between input and Actor Critic (Linear Hidden) - *linhid
- Agent with a nonlinear feedforward layer between input and Actor Critic (Nonlinear Hidden) - *hidden
- Agent with a Reservoir between input and Actor Critic (No Bump Reservoir) - *res
- Agent with a nonlinear feedforward layer between input and Actor Critic and a ring attractor for working memory (Hidden+Bump) - *hidden_bump
- Agent with a Reservoir between input and Actor Critic and a ring attractor for working memory (Reservoir+Bump) - *res_bump
- Agent with a nonlinear feedforward layer between input and Actor Critic and trained by backpropagation (A2C) - *hidden_a2c


## Requirements

System information

- OS: Windows 10

- Python version: Python 3.7.9
- matplotlib==3.4.3
- numpy==1.19.5
- pandas==1.3.3
- scipy==1.4.1
- tensorflow==2.5.0


```setup
pip install requirements.txt
```

## Single location training & evaluation

To run each agent described in the paper in the single location task, set working directory to ./1pa

To run Control agent:
```train
python 1pa_control.py
```

To train Classic agent:
```train
python 1pa_classic.py
```
To train Expanded Classic agent:
```train
python 1pa_expclas.py
```
To train Linear Hidden agent:
```train
python 1pa_linhid.py
```
To train Nonlinear Hidden agent:
```train
python 1pa_hidden.py
```

To train Reservoir agent:
```train
python 1pa_res.py
```

To train A2C agent:
```train
python 1pa_hidden_a2c.py
```

## six paired association training & evaluation

To run each agent in the six paired association task, set working directory to ./6pa

To run Control agent:
```train
python 6pa_control.py
```

To train Classic agent:
```train
python 6pa_classic.py
```
To train Expanded Classic agent:
```train
python 6pa_expclas.py
```
To train Linear Hidden agent:
```train
python 6pa_linhid.py
```
To train Nonlinear Hidden agent:
```train
python 6pa_hidden.py
```

To train Reservoir agent:
```train
python 6pa_res.py
```

To train A2C agent:
```train
python 6pa_hidden_a2c.py
```


## 16 paired association training & evaluation
To run each agent in the 16 PA task, set working directory to ./16pa

To train Nonlinear Hidden agent:
```train
python 16pa_hidden.py
```

To train A2C agent:
```train
python 16pa_hidden_a2c.py
```
## Working memory + Six paired association training & evaluation

To run each agent in the six paired association task with transient cue, set working directory to ./wkm

To train Classic agent:
```train
python wkm_classic.py
```
To train Expanded Classic agent:
```train
python wkm_expclas.py
```
To train Linear Hidden agent:
```train
python wkm_linhid.py
```
To train Nonlinear Hidden agent:
```train
python wkm_hidden.py
```

To train Nonlinear Hidden agent with ring attractor:
```train
python wkm_hidden_bump.py
```

To train Reservoir agent:
```train
python wkm_res.py
```

To train Reservoir agent:
```train
python wkm_res_bump.py
```

## Training details

Since the outcome of the paper is to demonstrate gradual learning of multiple paired associations, there are no pretrained agents. The learning potential of each agent can be observed by running the respective scripts.
Training for each agent takes about 1 hour for both single and multiple paired association task.

General agent hyperparameters can be found in get_default_hp function in ./backend_scripts/utils.py. Specific hyperparameters can be found in each *.py script.

E.g. if you would want to change the timestep from 100ms to 5ms, set hp['tstep'] = 5 instead of 100.

Each code generates a figure that is saved to the working directory. 


## Results

Our agents achieve the following performance for single location task :

- Latency & Time spent at initial and displaced single location by all agents:

![Fig1_small](https://user-images.githubusercontent.com/35286288/147554709-1907c093-2548-42ca-9803-0d2da2f05542.png)

- Only the Actor Critic with nonlinear hidden layer is able to learn the multiple paired association task :

![Fig3_small](https://user-images.githubusercontent.com/35286288/147554723-016a6bcf-ed17-4f54-a842-4f4f718eabe7.png)

## Contributing
Please cite the relevant work if the code is used for academic purposes.
