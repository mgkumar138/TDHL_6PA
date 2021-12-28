from backend_scripts.pa_task import multiplepa_script
from backend_scripts.utils import get_default_hp
import time as dt


if __name__ == '__main__':

    hp = get_default_hp(task='wkm',platform='laptop')

    hp['controltype'] = 'reservoir'  # expand, hidden, classic
    hp['tstep'] = 100  # deltat
    hp['btstp'] = 1
    hp['trsess'] = 20
    hp['evsess'] = 2
    hp['time'] = 600  # Tmax seconds
    hp['savefig'] = True
    hp['savevar'] = False
    hp['savegenvar'] = False

    ''' Hidden parameters '''
    hp['nrnn'] = 8192  # number of hidden units
    hp['resact'] = 'phia'
    hp['resrecact'] = 'tanh'
    hp['rtau'] = 150
    hp['chaos'] = 1.5
    hp['cp'] = [1,1]
    hp['resns'] = 0.025
    hp['fbsz'] = 0
    hp['sparsity'] = 3

    ''' Other Model parameters '''
    hp['lr'] = 0.00001
    hp['vscale'] = 1
    hp['usebump'] = False

    # First 5 seconds: place cell activity & action update switched off, sensory cue given
    # After 5 seconds: place cell activity & action update switched on, sensory cue silenced
    hp['workmem'] = True

    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = 'control_{}_{}_{}ha_{}wkm_{}bump_{}e_{}R_{}lr_{}dt_b{}_{}'.format(
        hp['task'], hp['controltype'],hp['hidact'],hp['workmem'],hp['usebump'], hp['eulerm'],hp['Rval'],
        hp['lr'], hp['tstep'],hp['btstp'],dt.monotonic())

    totlat, totdgr, totpi, mvpath, allw, alldyn = multiplepa_script(hp)
