from backend_scripts.pa_task import multiplepa_script
import time as dt
from backend_scripts.utils import get_default_hp

if __name__ == '__main__':

    hp = get_default_hp(task='6pa',platform='laptop')

    hp['controltype'] = 'classic'  # expand, hidden, classic
    hp['tstep'] = 100  # deltat
    hp['trsess'] = 100
    hp['btstp'] = 1
    hp['time'] = 600  # Tmax seconds
    hp['savefig'] = True
    hp['savevar'] = False
    hp['savegenvar'] = False

    ''' Hidden parameters '''
    hp['nhid'] = 67  # number of hidden units
    hp['hidact'] = False

    ''' Other Model parameters '''
    hp['lr'] = 0.001  # 0.001
    hp['eulerm'] = 1

    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = '{}_{}_{}ha_{}e_{}lr_{}dt_b{}_{}'.format(
        hp['task'], hp['controltype'],hp['hidact'], hp['eulerm'], hp['lr'], hp['tstep'],hp['btstp'],dt.monotonic())

    totlat, totdgr, totpi, mvpath, allw, alldyn = multiplepa_script(hp)
