import numpy as np
import matplotlib.pyplot as plt
from utils import find_cue, saveload, get_default_hp, savefigformats
import time as dt
from maze_env import MultiplePAs
import multiprocessing as mp
from functools import partial

pithres = 40


def run_multiple_expt(b, mtype, env, hp, agent, sessions, useweight=None, nocue=None, noreward=None):
    lat = np.zeros(sessions)
    dgr = []
    env.make(mtype=mtype, nocue=nocue, noreward=noreward)

    if useweight:
        agent.model.set_weights(useweight)

    for t in range(sessions * len(env.rlocs)):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.ac.reset()
        agent.cri_reset()
        value = agent.vstate
        rfr = agent.rstate
        rho = agent.ac.qstate

        if t % len(env.rlocs) == 0:
            sesslat = []

        while not done:
            if env.rendercall:
                env.render()

            # Plasticity switched off when trials are non-rewarded
            if t in env.nort and t in env.noct:
                plastic = False
            else:
                plastic = True

            # plasticity using Forward euler
            if hp['eulerm'] == 1:
                rpe, value = agent.learn(pre=rfr, post=rho, R=reward, v=value,plasticity=plastic,s1=state,cue_r1_fb=cue)

            # Pass coordinates to Place Cell & LCM to get actor & critic values
            rfr, q, _ = agent.act(state=state, cue_r_fb=cue)

            # Convolve actor dynamics & Action selection
            action, rho = agent.ac.move(q)

            # Use action on environment, ds4r: distance from reward
            state, cue, reward, done, ds4r = env.step(action)

            # plasticity using Backward euler
            if hp['eulerm'] == 0:
                rpe, value = agent.learn(pre=rfr, post=rho, R=reward, v=value, plasticity=plastic, s1=state,
                                         cue_r1_fb=cue)

            if done:
                break

        # if non-rewarded trial, save entire path trajectory & store visit rate
        if t in env.nort:
            sesslat.append(np.nan)
            dgr.append(env.dgr)
        else:
            sesslat.append(env.i)

        if (t + 1) % len(env.rlocs) == 0:
            lat[((t + 1) // len(env.rlocs)) - 1] = np.mean(sesslat)

        if hp['platform'] == 'laptop' or b == 0:
            # Trial information
            print('T {} | C {} | S {} | TD {:4.3f} | D {:4.3f} | st {} | Dgr {}'.format(
                t, find_cue(env.cue), env.i // (1000 // env.tstep), rpe.numpy()[0][0], ds4r, env.startpos[0], dgr))

            # Session information
            if (t + 1) % len(env.rlocs) == 0:
                print('################## {} Session {}/{}, Avg Steps {:5.1f}, PI {} ################'.format(
                    mtype, (t + 1) // len(env.rlocs), sessions, lat[((t + 1) // len(env.rlocs)) - 1], env.sessr))

    # get mean visit rate
    # evaluation sessions
    sesspi = np.sum(np.array(dgr) > pithres)
    dgr = np.mean(dgr)

    if hp['platform'] == 'server':
        print('Agent {} {} training dig rate: {}'.format(b, mtype, dgr))

    return lat, dgr, sesspi


def multiplepa_script(hp, b):
    import tensorflow as tf
    from model import SimpleAgent

    print('Agent {} started training ...'.format(b))

    trsess = hp['trsess']

    # Create nonrewarded probe trial index
    nonrp = [trsess]  # sessions that are non-rewarded probe trials
    env = MultiplePAs(hp)
    env.make(mtype='{}PA'.format(hp['npa']), noreward=nonrp)

    # Start experiment
    tf.compat.v1.reset_default_graph()
    start = dt.time()
    agent = SimpleAgent(env=env, hp=hp)

    # Start Training
    lat, dgr, pi = run_multiple_expt(b, 'train', env, hp, agent, trsess, noreward=nonrp)

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, pi


def main_script(hp):
    print('Number of PA: {}, thres sparsity: {}'.format(hp['npa'], hp['sparsity']))
    btstp = hp['btstp']

    pool = mp.Pool(processes=hp['cpucount'])
    x = pool.map(partial(multiplepa_script, hp), np.arange(btstp))

    pool.close()
    pool.join()

    totlat = np.zeros([btstp, int(hp['trsess'])])
    totdgr = np.zeros([btstp])
    totpi = np.zeros_like(totdgr)

    for b in range(btstp):
        totlat[b], totdgr[b], totpi[b] = x[b]

    return np.mean(totlat, axis=0), np.mean(totdgr, axis=0), np.mean(totpi, axis=0), \
           np.std(totlat, axis=0), np.std(totdgr, axis=0), np.std(totpi, axis=0)


if __name__ == '__main__':

    hp = get_default_hp(task='6pa', platform='server')

    hp['controltype'] = 'hidden'  # expand, hidden, classic
    hp['tstep'] = 100  # deltat
    hp['trsess'] = 101
    hp['btstp'] = 24
    hp['time'] = 600  # Tmax seconds
    hp['savefig'] = False
    hp['savevar'] = False

    ''' Hidden parameters '''
    hp['taug'] = 2000  # number of hidden units
    hp['hidact'] = 'reluthres'
    totparams = np.linspace(-5,5,11)  # sparsity

    ''' Other Model parameters '''
    hp['lr'] = 0.00001

    # First 30seconds: place cell activity & action update switched off, sensory cue given
    # After 30seconds: place cell activity & action update switched on, sensory cue silenced
    hp['workmem'] = False

    hp['render'] = False  # visualise movement trial by trial

    # datafile = glob.glob('./hyperparam/actfunc_*')[0]
    # [totlat, totdgr, totpi, stdlat, stddgr, stdpi] = saveload('load', datafile[:-7],1)

    hp['exptname'] = 'thres_16pa_{}_{}_{}ha_{}lr_{}wkm_{}3fc_{}dt_b{}_{}'.format(
        hp['task'], hp['controltype'], hp['hidact'], hp['lr'], hp['workmem'],
        hp['c3fr'], hp['tstep'], hp['btstp'], dt.monotonic())

    totpi = np.zeros([len(totparams)])
    totdgr = np.zeros([len(totparams)])
    totlat = np.zeros([len(totparams), int(hp['trsess'])])
    stdpi = np.zeros([len(totparams)])
    stddgr = np.zeros([len(totparams)])
    stdlat = np.zeros([len(totparams), int(hp['trsess'])])

    for n, param in enumerate(totparams):
        hp['sparsity'] = param
        hp['npa'] = 16

        totlat[n], totdgr[n], totpi[n], stdlat[n], stddgr[n], stdpi[n] = main_script(hp)

        saveload('save', '{}'.format(hp['exptname']), [totlat, totdgr, totpi, stdlat, stddgr, stdpi])

    # plot
    f1, (ax1, ax11) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.errorbar(x=totparams, y= totdgr, yerr= stddgr, color='deeppink')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, 100.5)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Mean Visit Ratio (%)')
    ax1.set_title('Visit Ratio after {} sessions'.format(hp['trsess'] -1))

    ax11.errorbar(x=totparams, y=totpi, yerr=stdpi, color='tab:green')
    ax11.spines['right'].set_visible(False)
    ax11.spines['top'].set_visible(False)
    ax11.set_ylim(0, 16.5)
    ax11.set_xlabel('Threshold')
    ax11.set_title('Number of PAs learnt with input expansion')
    ax11.set_ylabel('PAs with visit ratio > {}%'.format(pithres))

    f1.tight_layout()
    plt.show()

    savefigformats('16pa_thres')
