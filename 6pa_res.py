from pa_task import multiplepa_script
from utils import get_default_hp
import time as dt

if __name__ == '__main__':

    hp = get_default_hp(task='6pa',platform='laptop')

    hp['controltype'] = 'reservoir'  # expand, hidden, classic
    hp['tstep'] = 100  # deltat
    hp['btstp'] = 1
    hp['time'] = 600  # Tmax seconds
    hp['savefig'] = True
    hp['savevar'] = False
    hp['saveweight'] = False
    hp['savegenvar'] = False

    ''' Hidden parameters '''
    hp['nhid'] = 8192  # number of hidden units
    hp['hidact'] = 'relusparse'
    hp['sparsity'] = 3
    hp['resrecact'] = 'tanh'
    hp['rtau'] = 150
    hp['chaos'] = 1.5
    hp['cp'] = [1,1]
    hp['resns'] = 0.025
    hp['fbsz'] = 0

    ''' Other Model parameters '''
    hp['lr'] = 0.00001
    hp['eulerm'] = 1

    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = '{}_{}_{}ha_{}e_{}lr_{}dt_b{}_{}'.format(
        hp['task'], hp['controltype'],hp['hidact'], hp['eulerm'], hp['lr'], hp['tstep'],hp['btstp'],dt.monotonic())

    totlat, totdgr, totpi, mvpath, allw, alldyn = multiplepa_script(hp)
