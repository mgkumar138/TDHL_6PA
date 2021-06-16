from backend_scripts.utils import get_default_hp
import time as dt
from backend_scripts.pa_task import multiplepa_script


if __name__ == '__main__':
    '''
    Training an agent with a single nonlinear hidden layer to learn 6PAs using A2C.
    2D state information is passed to place cells and concatenated with cue. 
    Agent has a nonlinear hidden layer whose activity is passed to the actor and critic.
    State is continuous while action is discrete.
    Training of weights is by backpropagation of error signals to determine the gradients.
    Loss function is defined using the Advantage Actor Critic (A2C) algorithm.
    '''
    hp = get_default_hp(task='6pa',platform='laptop')

    hp['controltype'] = 'hidden'  # expand, hidden, classic
    hp['tstep'] = 100  # deltat = 100ms ** A2C algorithm tested only at dt = 100ms
    hp['trsess'] = 100
    hp['btstp'] = 1
    hp['time'] = 600  # Tmax seconds
    hp['savefig'] = True
    hp['savevar'] = False
    hp['savegenvar'] = False

    ''' Hidden parameters '''
    hp['nhid'] = 8192  # number of hidden units
    hp['hidact'] = 'phia'
    hp['K'] = None
    hp['sparsity'] = 3

    ''' Other Model parameters '''
    hp['lr'] = 0.000035
    hp['taug'] = 10000
    hp['actalpha'] = 1/4  # to smoothen action taken by agent
    hp['maxspeed'] = 0.07  # step size per 100ms

    hp['entbeta'] = -0.001
    hp['valalpha'] = 0.5

    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = 'A2C_{}tg_{}av_{}be_{}_{}_{}n_{}ha_{}lr_{}dt_b{}_{}'.format(hp['taug'],hp['valalpha'],hp['entbeta'],
        hp['task'], hp['controltype'],hp['nhid'], hp['hidact'],hp['lr'], hp['tstep'],hp['btstp'],dt.monotonic())

    totlat, totdgr, totpi, mvpath, trw, alldyn = multiplepa_script(hp)