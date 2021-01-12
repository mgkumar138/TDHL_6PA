from backend_scripts.pa_task import multiplepa_script
import time as dt
from backend_scripts.utils import get_default_hp

if __name__ == '__main__':

    hp = get_default_hp(task='6pa',platform='laptop')

    hp['controltype'] = 'hidden'  # expand, hidden, classic
    hp['tstep'] = 100  # deltat
    hp['trsess'] = 100
    hp['btstp'] = 1
    hp['time'] = 600  # Tmax seconds
    hp['savefig'] = True
    hp['savevar'] = False
    hp['savegenvar'] = False

    ''' Hidden parameters '''
    hp['nhid'] = 8192  # number of hidden units
    hp['hidact'] = 'relu'
    hp['K'] = None
    hp['sparsity'] = 0

    ''' Other Model parameters '''
    hp['lr'] = 0.00001
    hp['eulerm'] = 1  # Forward Euler approx for TD error
    hp['taug'] = 2000  # TD error time constant

    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = '{}_{}_{}ha_{}e_{}lr_{}dt_b{}_{}'.format(
        hp['task'], hp['controltype'],hp['hidact'], hp['eulerm'], hp['lr'], hp['tstep'],hp['btstp'],dt.monotonic())

    totlat, totdgr, totpi, mvpath, allw, alldyn = multiplepa_script(hp)

