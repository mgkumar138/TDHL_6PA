from pa_task import multiplepa_script
from utils import get_default_hp
import time as dt


if __name__ == '__main__':

    hp = get_default_hp(task='wkm',platform='laptop')

    hp['controltype'] = 'reservoir'  # expand, hidden, classic
    hp['tstep'] = 100  # deltat
    hp['btstp'] = 1
    hp['trsess'] = 20
    hp['evsess'] = 2
    hp['time'] = 600  # Tmax seconds
    hp['savefig'] = False
    hp['savevar'] = False
    hp['saveweight'] = False
    hp['savegenvar'] = True

    ''' Hidden parameters '''
    hp['nrnn'] = 8192  # number of hidden units
    hp['resact'] = 'relusparse'
    hp['resrecact'] = 'tanh'
    hp['rtau'] = 150
    hp['chaos'] = 1.5
    hp['cp'] = [1,1]
    hp['resns'] = 0.025
    hp['fbsz'] = 41
    hp['sparsity'] = 2

    ''' Other Model parameters '''
    hp['lr'] = 0.00001
    hp['vscale'] = 1
    hp['usebump'] = False

    # First 30seconds: place cell activity & action update switched off, sensory cue given
    # After 30seconds: place cell activity & action update switched on, sensory cue silenced
    hp['workmem'] = True

    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = 'tse_control_{}_{}_{}ha_{}wkm_{}bump_{}e_{}v_{}lr_{}dt_b{}_{}'.format(
        hp['task'], hp['controltype'],hp['hidact'],hp['workmem'],hp['usebump'], hp['eulerm'],hp['vscale'],
        hp['lr'], hp['tstep'],hp['btstp'],dt.monotonic())

    totlat, totdgr, totpi, mvpath, allw, alldyn = multiplepa_script(hp)

