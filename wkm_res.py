from pa_task import multiplepa_script
from utils import get_default_hp
import time as dt

if __name__ == '__main__':

    hp = get_default_hp(task='wkm',platform='gpu')

    hp['controltype'] = 'reservoir'  # expand, hidden, classic
    hp['tstep'] = 100  # deltat
    hp['btstp'] = 1
    hp['trsess'] = 20
    hp['evsess'] = 2
    hp['time'] = 600  # Tmax seconds
    hp['savefig'] = True
    hp['savevar'] = False
    hp['saveweight'] = False
    hp['savegenvar'] = False

    ''' Hidden parameters '''
    hp['nrnn'] = 8192  # number of hidden units
    hp['hidact'] = 'relusparse'
    hp['resrecact'] = 'tanh'
    hp['rtau'] = 150
    hp['chaos'] = 1.5
    hp['cp'] = [1,1]
    hp['resns'] = 0.025
    hp['fbsz'] = 41
    hp['sparsity'] = 2

    ''' Bump parameter '''
    hp['nwm'] = 54
    hp['psi'] = 300
    hp['bumpw-'] = -0.75
    hp['btau'] = 150
    hp['bact'] = 'relu'
    hp['usebump'] = True

    ''' Other Model parameters '''
    hp['lr'] = 0.00001
    hp['eulerm'] = 1

    # First 30seconds: place cell activity & action update switched off, sensory cue given
    # After 30seconds: place cell activity & action update switched on, sensory cue silenced
    hp['workmem'] = True
    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = 'tse_{}_{}_{}ha_{}wkm_{}bump_{}e_{}v_{}lr_{}dt_b{}_{}'.format(
        hp['task'], hp['controltype'],hp['hidact'],hp['workmem'],hp['usebump'], hp['eulerm'],hp['vscale'],
        hp['lr'], hp['tstep'],hp['btstp'],dt.monotonic())

    totlat, totdgr, totpi, mvpath, allw, alldyn = multiplepa_script(hp)
