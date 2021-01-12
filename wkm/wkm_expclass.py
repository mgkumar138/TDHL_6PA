from backend_scripts.pa_task import multiplepa_script
import time as dt
from backend_scripts.utils import get_default_hp

if __name__ == '__main__':

    hp = get_default_hp(task='wkm',platform='laptop')

    hp['controltype'] = 'expand'  # expand, hidden, classic
    hp['tstep'] = 100  # deltat
    hp['btstp'] = 1
    hp['trsess'] = 20
    hp['evsess'] = 2
    hp['time'] = 600  # Tmax seconds
    hp['savefig'] = True
    hp['savevar'] = False
    hp['savegenvar'] = False

    ''' Hidden parameters '''
    hp['nhid'] = 8192  # number of hidden units
    hp['hidact'] = False

    ''' Bump parameter '''
    hp['nwm'] = 54
    hp['bact'] = 'relu'
    hp['usebump'] = True

    ''' Other Model parameters '''
    hp['lr'] = 0.00001

    # First 5 seconds: place cell activity & action update switched off, sensory cue given
    # After 5 seconds: place cell activity & action update switched on, sensory cue silenced
    hp['workmem'] = True

    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = '{}_{}_{}ha_{}wkm_{}bump_{}e_{}R_{}lr_{}dt_b{}_{}'.format(
        hp['task'], hp['controltype'],hp['hidact'],hp['workmem'],hp['usebump'], hp['eulerm'],hp['Rval'],
        hp['lr'], hp['tstep'],hp['btstp'],dt.monotonic())

    totlat, totdgr, totpi, mvpath, allw, alldyn = multiplepa_script(hp)

