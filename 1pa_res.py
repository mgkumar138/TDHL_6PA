from singlepa import singlepa_script
import time as dt
from utils import get_default_hp

if __name__ == '__main__':

    hp = get_default_hp(task='1pa',platform='laptop')

    hp['controltype'] = 'reservoir'  # expand, hidden, classic, reservoir
    hp['tstep'] = 100  # deltat
    hp['btstp'] = 3
    hp['time'] = 300  # Tmax seconds
    hp['savefig'] = True
    hp['savevar'] = False
    hp['saveweight'] = False
    hp['savegenvar'] = False

    ''' Hidden parameters '''
    hp['nhid'] = 1024  # number of hidden units
    hp['hidact'] = 'relusparse'
    hp['sparsity'] = 2
    hp['resrecact'] = 'tanh'
    hp['rtau'] = 150
    hp['chaos'] = 1.5
    hp['cp'] = [1,1]
    hp['resns'] = 0.025
    hp['fbsz'] = 41

    ''' Other Model parameters '''
    hp['lr'] = 0.0001  # 0.0001

    # First 30seconds: place cell activity & action update switched off, sensory cue given
    # After 30seconds: place cell activity & action update switched on, sensory cue silenced
    hp['workmem'] = False
    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = '{}_{}_{}ha_{}e_{}lr_{}dt_b{}_{}'.format(
        hp['task'], hp['controltype'],hp['hidact'], hp['eulerm'], hp['lr'], hp['tstep'],hp['btstp'],dt.monotonic())

    totlat, totdgr, diffw, mvpath, allw, alldyn = singlepa_script(hp)
