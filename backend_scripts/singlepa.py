import numpy as np
import matplotlib.pyplot as plt
from backend_scripts.utils import save_rdyn, find_cue, saveload
import time as dt
import pandas as pd
from backend_scripts.maze_env import Maze
import multiprocessing as mp
from functools import partial


def singlepa_script(hp):
    exptname = hp['exptname']
    btstp = hp['btstp']

    print(exptname)
    tt = hp['trsess'] * 6  # Number of session X number of trials per session
    et = hp['evsess'] * 6

    # store performance
    totlat = np.zeros([btstp, (tt + et * 7)])
    totdgr = np.zeros([btstp, 8, 3])
    diffw = np.zeros([btstp, 2, 7])  # bt, number of layers, modelcopy

    pool = mp.Pool(processes=hp['cpucount'])

    if hp['controltype'] == 'reservoir':
        x = pool.map(partial(main_single_res_expt, hp), np.arange(btstp))
    else:
        x = pool.map(partial(main_single_expt, hp),np.arange(btstp))

    pool.close()
    pool.join()

    # Start experiment
    for b in range(btstp):
        totlat[b], totdgr[b], diffw[b], mvpath, allw, alldyn = x[b]

    plt.figure(figsize=(15, 8))
    plt.gcf().text(0.01, 0.01, exptname, fontsize=12)
    plt.subplot(231)
    plt.title('Latency per Trial')
    plt.errorbar(x=np.arange(totlat.shape[1]), y=np.mean(totlat, axis=0), yerr=np.std(totlat, axis=0))
    plt.plot(np.mean(totlat, axis=0), 'k', linewidth=3)

    plt.subplot(232)
    df = pd.DataFrame(np.mean(totdgr, axis=0), columns=['2', '5', '10'], index=['Initial', 'Same', '1D', '2D', '3D', '4D', '5D', '6D'])
    ds = pd.DataFrame(np.std(totdgr, axis=0), columns=['2', '5', '10'], index=['Initial', 'Same', '1D', '2D', '3D', '4D', '5D', '6D'])
    df.plot.bar(rot=0, ax=plt.gca(), yerr=ds,legend=False)
    plt.axhline(y=np.mean(totdgr, axis=0)[0,0], color='r', linestyle='--')

    plt.subplot(233)
    df = pd.DataFrame(np.mean(diffw[:,-2:], axis=0), columns=['1PA', '1DPA', '2DPA','3DPA', '4DPA','5DPA', '6DPA'], index=['Critic', 'Actor'])
    ds = pd.DataFrame(np.std(diffw[:,-2:], axis=0), columns=['1PA', '1DPA', '2DPA','3DPA', '4DPA','5DPA', '6DPA'], index=['Critic', 'Actor'])
    df.plot.bar(rot=0, ax=plt.gca(), yerr=ds)

    # create environment
    env = Maze(hp)

    col = ['b', 'g', 'r', 'y', 'm', 'k']
    for i, k, j in zip(np.arange(4, 7), [mvpath[1,2], mvpath[3,2], mvpath[7,2]],
                          ['1pa', '2dpa', '6dpa']):
        plt.subplot(2, 3, i)
        plt.title('{} steps {}'.format(j, len(k)))
        env.make(j)
        for pt in range(len(mvpath[2])):
            plt.plot(np.array(k[pt])[:, 0], np.array(k[pt])[:, 1], col[pt], alpha=0.5)
            circle = plt.Circle(env.rlocs[pt], env.rrad, color=col[i-3])
            plt.gcf().gca().add_artist(circle)
        plt.axis((-env.au / 2, env.au / 2, -env.au / 2, env.au / 2))

    if hp['savefig']:
        plt.savefig('./1pa/Fig/fig_{}.png'.format(exptname))
    if hp['savegenvar']:
        saveload('save', './1pa/Data/genvars_{}_b{}_{}'.format(exptname,btstp, dt.monotonic()), [totlat, totdgr, diffw])
    print(exptname)

    plt.tight_layout()

    return totlat, totdgr, diffw, mvpath, allw, alldyn


''' Feedforward models '''


def run_single_expt(b, mtype, env, hp, agent, alldyn, trials, useweight=None, nocue=None, noreward=None):
    lat = np.zeros(trials)
    dgr = []
    mvpath = np.zeros((3, 6, env.normax, 2))

    env.make(mtype=mtype, nocue=nocue, noreward=noreward)

    if useweight:
        agent.model.set_weights(useweight)

    for t in range(trials):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.ac.reset()
        agent.cri_reset()
        wtrack = []
        value = agent.vstate
        rfr = agent.rstate
        rho = agent.ac.qstate

        while not done:
            if env.rendercall:
                env.render()

            # Plasticity switched off during non-rewarded probe trials
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
                rpe, value = agent.learn(pre=rfr, post=rho, R=reward, v=value,plasticity=plastic,s1=state,cue_r1_fb=cue)

            # save lsm & actor dynamics for analysis
            if hp['savevar']:
                if t in env.nort:
                    save_rdyn(alldyn[0], mtype, t, env.startpos, env.cue, rfr)
                    save_rdyn(alldyn[1], mtype, t, env.startpos, env.cue, rho)
                    save_rdyn(alldyn[2], mtype, t, env.startpos, env.cue, value)
                    save_rdyn(alldyn[3], mtype, t, env.startpos, env.cue, agent.tderr)
                else:
                    wtrack.append([agent.dwc,agent.dwa])

            if done:
                break

        # if non-rewarded trial, save entire path trajectory & store visit rate
        if t in env.nort:
            lat[t] = np.nan
            dgr.append(env.dgr)
            sid = np.argmax(np.array(noreward) == (t // 6) + 1)
            mvpath[sid, t%6] = env.tracks[:env.normax]
        else:
            lat[t] = env.i
            alldyn[4].append(np.sum(np.array(wtrack), axis=0))

        if hp['platform'] =='laptop' or b==0:
            # Trial information
            print('T {} | C {} | S {} | TD {:4.3f} | D {:4.3f} | Dgr {}'.format(
                t, find_cue(env.cue), env.i // (1000 // env.tstep), rpe.numpy()[0][0], ds4r, np.round(dgr)))

    # get mean visit rate
    if len(noreward) > 1:
        # training session
        dgr = np.mean(np.array(dgr).reshape(len(noreward), 6), axis=1)
    else:
        # evaluation sessions
        dgr = np.mean(dgr)

    mdlw = agent.model.get_weights()

    if hp['platform'] == 'server':
        print('Agent {} {} training dig rate: {}'.format(b, mtype, dgr))
    return lat, mvpath, mdlw, dgr


def main_single_expt(hp,b):
    import tensorflow as tf
    from backend_scripts.model import SimpleAgent

    print('Agent {} started training ...'.format(b))
    exptname = hp['exptname']

    # create environment
    env = Maze(hp)

    tt = hp['trsess'] * 6  # Number of session X number of trials per session
    et = hp['evsess'] * 6

    lat = np.zeros(tt + et * 7)
    dgr = np.zeros([8, 3])
    diffw = np.zeros([2, 7])  # bt, number of layers, modelcopy
    nonr = [2, 5, 10]
    rdyn = {}
    qdyn = {}
    cdyn = {}
    tdyn = {}
    wtrk = []
    alldyn = [rdyn, qdyn, cdyn, tdyn, wtrk]
    mvpath = np.zeros([8, 3, 6, env.normax, 2])
    tf.compat.v1.reset_default_graph()
    start = dt.time()
    agent = SimpleAgent(hp=hp, env=env)

    # Start Training
    lat[:tt], mvpath[0], trw, dgr[0] = run_single_expt(b, '1train', env, hp, agent, alldyn, tt, noreward=nonr)

    # Start Evaluation
    lat[tt:(tt + et)], mvpath[1], w1pa, dgr[1] = run_single_expt(b, '1pa', env, hp, agent, alldyn, et, trw, noreward=nonr)
    lat[(tt + et):(tt + et * 2)], mvpath[2], w1dpa, dgr[2] = run_single_expt(b, '1dpa', env, hp, agent, alldyn, et, trw,
                                                                                      noreward=nonr)
    lat[(tt + et * 2):(tt + et * 3)], mvpath[3], w2dpa, dgr[3] = run_single_expt(b, '2dpa', env, hp, agent, alldyn, et,
                                                                                         trw, noreward=nonr)
    lat[(tt + et * 3):(tt + et * 4)], mvpath[4], w3dpa, dgr[4] = run_single_expt(b, '3dpa', env, hp, agent, alldyn, et,
                                                                                          trw, noreward=nonr)
    lat[(tt + et * 4):(tt + et * 5)], mvpath[5], w4dpa, dgr[5] = run_single_expt(b, '4dpa', env, hp, agent, alldyn, et,
                                                                                          trw, noreward=nonr)
    lat[(tt + et * 5):(tt + et * 6)], mvpath[6], w5dpa, dgr[6] = run_single_expt(b, '5dpa', env, hp, agent, alldyn, et,
                                                                                          trw, noreward=nonr)
    lat[(tt + et * 6):], mvpath[7], w6dpa, dgr[7] = run_single_expt(b, '6dpa', env, hp, agent, alldyn, et, trw,
                                                                             noreward=nonr)

    # Summarise weight change of layers
    for i, k in enumerate([w1pa, w1dpa, w2dpa, w3dpa, w4dpa, w5dpa, w6dpa]):
        for j in np.arange(-2,0):
            diffw[j, i] = np.sum(abs(k[j] - trw[j])) / np.size(k[j])

    allw = [trw, w1pa, w1dpa, w2dpa, w3dpa, w4dpa, w5dpa, w6dpa]

    if hp['savevar']:
        saveload('save', './1pa/Data/vars_{}_{}'.format(exptname, dt.monotonic()),
                 [rdyn, qdyn, cdyn, tdyn, wtrk, mvpath, lat, dgr, diffw])
    if hp['saveweight']:
        saveload('save', './1pa/Data/weights_{}_{}'.format(exptname, dt.monotonic()),
                 [trw, w1pa, w1dpa, w2dpa, w3dpa, w4dpa, w5dpa, w6dpa])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, diffw, mvpath, allw, alldyn


''' Reservoir model '''


def run_res_single_expt(b, mtype, env, hp, agent, alldyn, trials, useweight=None, nocue=None, noreward=None):
    lat = np.zeros(trials)
    dgr = []
    mvpath = np.zeros((3, 6, env.normax, 2))

    env.make(mtype=mtype, nocue=nocue, noreward=noreward)

    if useweight:
        agent.model.set_weights(useweight)

    for t in range(trials):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.ac.reset()
        agent.cri_reset()
        rstate = agent.rstate
        rfr = agent.rstate
        value = agent.vstate
        rho = agent.ac.qstate
        wtrack = []

        while not done:
            if env.rendercall == 'y':
                env.render()

            # Plasticity switched off when trials are non-rewarded & during cue presentation (60s)
            if t in env.nort and t in env.noct:
                plastic = False
            else:
                plastic = True

            # plasticity using Forward euler
            if hp['eulerm'] == 1:
                rpe, value = agent.learn(pre=rfr, post=rho, R=reward, v=value,plasticity=plastic,s1=state,
                                         cue_r1_fb=cue, h1=rstate)

            # Pass 2D coordinates to Place Cell & LCM with feedback to get actor & critic values
            q, rfr, rstate, _ = agent.act(state=state, cue_r_fb=cue, rstate=rstate)

            # Convolve actor dynamics & Action selection
            action, rho = agent.ac.move(q)

            # Use action on environment, ds4r: distance from reward
            state, cue, reward, done, ds4r = env.step(action)

            # plasticity using Backward euler
            if hp['eulerm'] == 0:
                rpe, value = agent.learn(pre=rfr, post=rho, R=reward, v=value,plasticity=plastic,s1=state,
                                         cue_r1_fb=cue, h1=rstate)

            # save lsm & actor dynamics for analysis
            if hp['savevar']:
                if t in env.nort:
                    save_rdyn(alldyn[0], mtype, t, env.startpos, env.cue, rfr)
                    save_rdyn(alldyn[1], mtype, t, env.startpos, env.cue, rho)
                    save_rdyn(alldyn[2], mtype, t, env.startpos, env.cue, value)
                    save_rdyn(alldyn[3], mtype, t, env.startpos, env.cue, agent.tderr)
                else:
                    wtrack.append([agent.dwc,agent.dwa])

            if done:
                break

        # if non-rewarded trial, save entire path trajectory & store visit rate
        if t in env.nort:
            lat[t] = np.nan
            dgr.append(env.dgr)
            sid = np.argmax(np.array(noreward) == (t // 6) + 1)
            mvpath[sid, t % 6] = env.tracks[:env.normax]
        else:
            lat[t] = env.i
            alldyn[4].append(np.sum(np.array(wtrack), axis=0))

        if hp['platform'] == 'laptop' or b ==0:
            # Trial information
            print('T {} | C {} | S {} | TD {:4.3f} | D {:4.3f} | Dgr {}'.format(
                t, find_cue(env.cue), env.i // (1000 // env.tstep), rpe.numpy()[0][0], ds4r,np.round(dgr)))

    # get mean visit rate
    if len(noreward) > 1:
        # training session
        dgr = np.mean(np.array(dgr).reshape(len(noreward), 6), axis=1)
    else:
        # evaluation sessions
        dgr = np.mean(dgr)

    mdlw = agent.model.get_weights()

    if hp['platform'] == 'server':
        print('Agent {} {} training dig rate: {}'.format(b, mtype, dgr))
    return lat, mvpath, mdlw, dgr


def main_single_res_expt(hp,b):
    import tensorflow as tf
    from backend_scripts.model import LSMAgent

    print('Agent {} started training ...'.format(b))
    exptname = hp['exptname']

    # create environment
    env = Maze(hp)

    tt = hp['trsess'] * 6  # Number of session X number of trials per session
    et = hp['evsess'] * 6

    lat = np.zeros(tt + et * 7)
    dgr = np.zeros([8, 3])
    diffw = np.zeros([2, 7])  # bt, number of layers, modelcopy
    nonr = [2, 5, 10]
    rdyn = {}
    qdyn = {}
    cdyn = {}
    tdyn = {}
    wtrk = []
    alldyn = [rdyn, qdyn, cdyn, tdyn, wtrk]
    mvpath = np.zeros([8, 3, 6, env.normax, 2])
    tf.compat.v1.reset_default_graph()
    start = dt.time()
    agent = LSMAgent(hp=hp, env=env)

    # Start Training
    lat[:tt], mvpath[0], trw, dgr[0] = run_res_single_expt(b, '1train', env, hp, agent, alldyn, tt, noreward=nonr)

    # Start Evaluation
    lat[tt:(tt + et)], mvpath[1], w1pa, dgr[1] = run_res_single_expt(b, '1pa', env, hp, agent, alldyn, et, trw, noreward=nonr)
    lat[(tt + et):(tt + et * 2)], mvpath[2], w1dpa, dgr[2] = run_res_single_expt(b, '1dpa', env, hp, agent, alldyn, et, trw,
                                                                                      noreward=nonr)
    lat[(tt + et * 2):(tt + et * 3)], mvpath[3], w2dpa, dgr[3] = run_res_single_expt(b, '2dpa', env, hp, agent, alldyn, et,
                                                                                         trw, noreward=nonr)
    lat[(tt + et * 3):(tt + et * 4)], mvpath[4], w3dpa, dgr[4] = run_res_single_expt(b, '3dpa', env, hp, agent, alldyn, et,
                                                                                          trw, noreward=nonr)
    lat[(tt + et * 4):(tt + et * 5)], mvpath[5], w4dpa, dgr[5] = run_res_single_expt(b, '4dpa', env, hp, agent, alldyn, et,
                                                                                          trw, noreward=nonr)
    lat[(tt + et * 5):(tt + et * 6)], mvpath[6], w5dpa, dgr[6] = run_res_single_expt(b, '5dpa', env, hp, agent, alldyn, et,
                                                                                          trw, noreward=nonr)
    lat[(tt + et * 6):], mvpath[7], w6dpa, dgr[7] = run_res_single_expt(b, '6dpa', env, hp, agent, alldyn, et, trw,
                                                                             noreward=nonr)

    # Summarise weight change of layers
    for i, k in enumerate([w1pa, w1dpa, w2dpa, w3dpa, w4dpa, w5dpa, w6dpa]):
        for j in np.arange(-2,0):
            diffw[j, i] = np.sum(abs(k[j] - trw[j])) / np.size(k[j])

    allw = [trw, w1pa, w1dpa, w2dpa, w3dpa, w4dpa, w5dpa, w6dpa]

    if hp['savevar']:
        saveload('save', './1pa/Data/vars_{}_{}'.format(exptname, dt.monotonic()),
                 [rdyn, qdyn, cdyn, tdyn, wtrk, mvpath, lat, dgr, diffw])
    if hp['saveweight']:
        saveload('save', './1pa/Data/weights_{}_{}'.format(exptname, dt.monotonic()),
                 [trw, w1pa, w1dpa, w2dpa, w3dpa, w4dpa, w5dpa, w6dpa])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, diffw, mvpath, allw, alldyn