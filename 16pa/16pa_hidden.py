import numpy as np
import matplotlib.pyplot as plt
from backend_scripts.utils import find_cue, saveload, get_default_hp, save_rdyn, plot_maps
import time as dt
from backend_scripts.maze_env import MultiplePAs
import multiprocessing as mp
from functools import partial

pithres = 40


def run_multiple_expt(b, mtype, env, hp, agent, sessions,alldyn, useweight=None, nocue=None, noreward=None):
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
            if t in env.nort or t in env.noct:
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

            if t in env.nort:
                save_rdyn(alldyn[0], mtype, t, env.startpos, env.cue, rho)
                save_rdyn(alldyn[1], mtype, t, env.startpos, env.cue, value)
                save_rdyn(alldyn[2], mtype, t, env.startpos, env.cue, agent.tderr)

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

    qdyn = {}
    cdyn = {}
    tdyn = {}
    alldyn = [qdyn,cdyn,tdyn]

    # Start experiment
    tf.compat.v1.reset_default_graph()
    start = dt.time()
    agent = SimpleAgent(env=env, hp=hp)

    # Start Training
    lat, dgr, pi, mvpath = run_multiple_expt(b, 'train', env, hp, agent, trsess,alldyn, noreward=nonrp)

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, pi, mvpath, alldyn


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
        totlat[b], totdgr[b], totpi[b], mvpath, alldyn = x[b]

    return totlat, totdgr, totpi, mvpath, alldyn


if __name__ == '__main__':

    hp = get_default_hp(task='6pa', platform='laptop')

    hp['controltype'] = 'hidden'  # expand, hidden, classic
    hp['tstep'] = 100  # deltat
    hp['trsess'] = 101  # number of training sessions
    hp['btstp'] = 1  # number of runs
    hp['time'] = 600  # Tmax seconds
    hp['savefig'] = True
    hp['savevar'] = False
    hp['npa'] = 16  # learn 2 - 16 PAs
    hp['Rval'] = 1  # increase for faster convergence
    hp['constR'] = 0
    hp['eulerm'] = 1
    hp['maxspeed'] = 0.03

    ''' Hidden parameters '''
    hp['nhid'] = 8192  # number of hidden units ~ Expansion ratio = nhid/67
    hp['hidact'] = 'relu'  # phiA, phiB, relu, etc
    hp['sparsity'] = None  # Threshold
    hp['K'] = None  # Number of positive connections from all inputs (67) to each hidden unit
    hp['taug'] = 2000    # TD error time constant

    ''' Other Model parameters '''
    hp['lr'] = 0.00001

    hp['exptname'] = '16pa_{}_{}_{}n_{}ha_{}th_{}k_{}R_{}tg_{}lr_{}dt_b{}_{}'.format(
        hp['task'], hp['controltype'], hp['nhid'],hp['hidact'],  hp['sparsity'],hp['K'],hp['Rval'], hp['taug'],
        hp['lr'], hp['tstep'], hp['btstp'], dt.monotonic())
    print(hp['exptname'])

    totlat, totdgr, totpi, mvpath, alldyn = main_script(hp)

    # plot
    f1, (ax1, ax11) = plt.subplots(1, 2)
    f1.text(0.01, 0.01, hp['exptname'], fontsize=10)
    ax1.errorbar(x=np.arange(1,1+hp['trsess']),y=np.mean(totlat*hp['tstep']/1000,axis=0), yerr=np.std(totlat*hp['tstep']/1000,axis=0))
    ax1.set_title('16 PA latency')
    ax1.set_xlabel('Sessions')
    ax1.set_xlabel('Latency (s)')

    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, hp['npa']))
    env = MultiplePAs(hp)
    env.make(mtype='16PA')
    for pt in range(hp['npa']):
        ax11.plot(mvpath[pt, :, 0], mvpath[pt, :, 1], color=colors[pt], alpha=0.5)
        circle = plt.Circle(env.rlocs[pt], env.rrad, color=colors[pt])
        ax11.add_artist(circle)
    ax11.axis((-env.au / 2, env.au / 2, -env.au / 2, env.au / 2))
    ax11.set_aspect('equal', adjustable='box')

    f1.tight_layout()

    if hp['savefig']:
        f1.savefig('./Fig/fig_{}.png'.format(hp['exptname']))
    if hp['savevar']:
        saveload('save', './Data/genvars_{}'.format(hp['exptname']), [totlat, totdgr, totpi, mvpath, alldyn])

    npa=16
    qdyn = alldyn[0]
    cdyn = alldyn[1]
    nonrlen = 600
    bins = 15
    qfr = np.zeros([npa, nonrlen, 40])
    cfr = np.zeros([npa, nonrlen, 1])
    policy = np.zeros([2, bins, bins])
    newx = np.zeros([225,2])
    for i in range(15):
        st = i * 15
        ed = st + 15
        newx[st:ed, 0] = np.arange(15)
    for i in range(15):
        st = i * 15
        ed = st + 15
        newx[st:ed, 1] = i * np.ones(15)

    sess = list(cdyn.keys())
    for s in sess:
        try:
            c = int(s[-2:])
        except ValueError:
            c = int(s[-1])
        qfr[c - 1] = np.array(qdyn[s])[-nonrlen:]
        cfr[c - 1] = np.array(cdyn[s])[-nonrlen:]

    from backend_scripts.model import action_cells
    from scipy.stats import binned_statistic_2d
    actor = action_cells(hp)

    plt.figure(figsize=(10,8))
    plt.gcf().text(0.01, 0.01, hp['exptname'], fontsize=10)
    totmax8 = []
    for i in range(npa):
        qpolicy = np.matmul(actor.aj, qfr[i].T)
        policy[0] = binned_statistic_2d(mvpath[i,:, 0], mvpath[i, :, 1], qpolicy[0], bins=bins, statistic='sum')[0]
        policy[1] = binned_statistic_2d(mvpath[i,:, 0], mvpath[i,:, 1], qpolicy[1], bins=bins, statistic='sum')[0]
        ccells = binned_statistic_2d(mvpath[i,:, 0], mvpath[i, :,1], cfr[i,:, 0], bins=bins, statistic='mean')[0]
        policy = np.nan_to_num(policy)
        ccells = np.nan_to_num(ccells)

        totmax8.append(np.round(np.sum(ccells >= np.max(ccells) * 0.8) / ccells.size, 3))
        plt.subplot(4,4,i+1)
        im = plt.imshow(ccells.T,aspect='auto',origin='lower')
        plt.title('C{}, .8m {}'.format(i+1, totmax8[i]), fontsize=10)
        plt.quiver(newx[:, 1], newx[:, 0], policy[1].reshape(bins ** 2), policy[0].reshape(bins ** 2),
                         units='xy',color='w')
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        plt.xticks([], [])
        plt.yticks([], [])
        plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    print(np.mean(totmax8))

    plt.savefig('./Fig/fig_map{}_{}.png'.format(np.mean(totmax8), hp['exptname']))