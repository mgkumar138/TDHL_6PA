import numpy as np
import matplotlib.pyplot as plt
from backend_scripts.utils import find_cue, saveload, get_default_hp
import time as dt
from backend_scripts.maze_env import MultiplePAs
import multiprocessing as mp
from functools import partial

pithres = 40


def run_multiple_expt(b, mtype, env, hp, agent, sessions, useweight=None, nocue=None, noreward=None):
    lat = np.zeros(sessions)
    mvpath = np.zeros((16, env.normax, 2))
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
            mvpath[find_cue(cue)-1] = np.array(env.tracks)[:env.normax]
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

    return lat, dgr, sesspi, mvpath


def multiplepa_script(hp, b):
    import tensorflow as tf
    from backend_scripts.model import SimpleAgent

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
    lat, dgr, pi, mvpath = run_multiple_expt(b, 'train', env, hp, agent, trsess, noreward=nonrp)

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, pi, mvpath


def main_script(hp):
    print('Number of PA: {}, Activation function used: {}'.format(hp['npa'], hp['hidact']))
    btstp = hp['btstp']

    pool = mp.Pool(processes=hp['cpucount'])
    x = pool.map(partial(multiplepa_script, hp), np.arange(btstp))

    pool.close()
    pool.join()

    totlat = np.zeros([btstp, int(hp['trsess'])])
    totdgr = np.zeros([btstp])
    totpi = np.zeros_like(totdgr)

    for b in range(btstp):
        totlat[b], totdgr[b], totpi[b], mvpath = x[b]

    return totlat, totdgr, totpi, mvpath


if __name__ == '__main__':

    hp = get_default_hp(task='6pa', platform='laptop')

    hp['controltype'] = 'hidden'  # expand, hidden, classic
    hp['tstep'] = 100  # deltat
    hp['trsess'] = 101
    hp['btstp'] = 1
    hp['time'] = 600  # Tmax seconds
    hp['savefig'] = False
    hp['savevar'] = False
    hp['npa'] = 16

    ''' Hidden parameters '''
    hp['nhid'] = 8192  # number of hidden units ~ Expansion ratio = nhid/67
    hp['hidact'] = 'relu'  # relu threshold - relusparse, phi threshold - reluthres, relu - threshold = 0
    hp['sparsity'] = 0  # Threshold
    hp['K'] = None  # Number of positive connections from all inputs (67) to each hidden unit
    hp['taug'] = 2000    # TD error time constant

    ''' Other Model parameters '''
    hp['lr'] = 0.00001

    hp['exptname'] = '16pa_{}_{}_{}n_{}ha_{}th_{}k_{}tg_{}lr_{}dt_b{}_{}'.format(
        hp['task'], hp['controltype'], hp['nhid'],hp['hidact'],  hp['sparsity'],hp['K'], hp['taug'],
        hp['lr'], hp['tstep'], hp['btstp'], dt.monotonic())

    totlat, totdgr, totpi, mvpath = main_script(hp)

    # plot
    f1, (ax1, ax11) = plt.subplots(1, 2)
    ax1.errorbar(x=np.arange(1,1+hp['trsess']),y=np.mean(totlat*hp['tstep']/1000,axis=0), yerr=np.std(totlat*hp['tstep']/1000,axis=0))
    ax1.set_title('16 PA latency')
    ax1.set_xlabel('Sessions')
    ax1.set_xlabel('Latency (s)')

    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, 16))
    env = MultiplePAs(hp)
    env.make(mtype='16PA')
    for pt in range(16):
        ax11.plot(mvpath[pt, :, 0], mvpath[pt, :, 1], colors[pt], alpha=0.5)
        circle = plt.Circle(env.rlocs[pt], env.rrad, color=colors[pt])
        ax11.add_artist(circle)
    ax11.axis((-env.au / 2, env.au / 2, -env.au / 2, env.au / 2))
    ax11.set_aspect('equal', adjustable='box')

    f1.tight_layout()

    if hp['savefig']:
        f1.savefig('./6pa/Fig/16pa_hidden.png')
    if hp['savevar']:
        saveload('save', './6pa/Data/{}'.format(hp['exptname']), [totlat, totdgr, totpi, mvpath])