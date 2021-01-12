import numpy as np
import matplotlib.pyplot as plt
from backend_scripts.utils import save_rdyn, find_cue, saveload, plot_dgr, plot_maps
import time as dt
from backend_scripts.maze_env import Maze
import multiprocessing as mp
from functools import partial

''' Classical & Hidden '''
pithres = 40

def multiplepa_script(hp):
    exptname = hp['exptname']
    btstp = hp['btstp']

    print(exptname)

    # store performance
    totlat = np.zeros([btstp, hp['trsess']])
    totdgr = np.zeros([btstp, 3])
    totpi = np.zeros_like(totdgr)
    scl = hp['trsess'] // 20  # scale number of sessions to Tse et al., 2007

    pool = mp.Pool(processes=hp['cpucount'])

    if hp['usebump']:
        if hp['controltype'] == 'reservoir':
            x = pool.map(partial(wkm_res_multiplepa_expt, hp), np.arange(btstp))
        else:
            x = pool.map(partial(wkm_multiplepa_expt, hp),np.arange(btstp))
    elif hp['usebump'] is False:
        if hp['controltype'] == 'reservoir':
            x = pool.map(partial(main_res_multiplepa_expt, hp), np.arange(btstp))
        else:
            x = pool.map(partial(main_multiplepa_expt, hp),np.arange(btstp))
    else:
        print('Error in task usebump: {}, task: {}'.format(hp['task'],hp['usebump']))
        breakpoint()

    pool.close()
    pool.join()

    # Start experiment
    for b in range(btstp):
        totlat[b], totdgr[b], totpi[b], mvpath, trw, alldyn = x[b]

    plt.figure(figsize=(15, 8))
    plt.gcf().text(0.01, 0.01, exptname, fontsize=10)
    plt.subplot(231)
    plt.title('Latency')
    plt.errorbar(x=np.arange(totlat.shape[1]), y =np.mean(totlat, axis=0), yerr=np.std(totlat,axis=0), marker='s')
    #plt.plot(np.mean(totlat,axis=0),linewidth=3)

    plot_dgr(totdgr, scl, 232, 6)

    plot_maps(alldyn,mvpath, hp, 233)

    env = Maze(hp)
    env.make('train')
    col = ['b', 'g', 'r', 'y', 'm', 'k']
    for i, k, j in zip(np.arange(4, 7), [mvpath[0], mvpath[1], mvpath[2]],
                             ['PS1', 'PS2', 'PS3']):
        plt.subplot(2, 3, i)
        plt.title('{}'.format(j))
        for pt in range(len(mvpath[2])):
            plt.plot(np.array(k[pt])[:, 0], np.array(k[pt])[:, 1], col[pt], alpha=0.5)
            circle = plt.Circle(env.rlocs[pt], env.rrad, color=col[pt])
            plt.gcf().gca().add_artist(circle)
        plt.axis((-env.au / 2, env.au / 2, -env.au / 2, env.au / 2))

    print(exptname)
    plt.tight_layout()

    if hp['savefig']:
        plt.savefig('./Fig/fig_{}.png'.format(exptname))
    if hp['savegenvar']:
        saveload('save', './Data/genvars_{}_b{}_{}'.format(exptname, btstp, dt.time()),
                 [totlat, totdgr, totpi])

    return totlat, totdgr, totpi, mvpath, trw, alldyn

''' Feedforwad models'''

def run_multiple_expt(b,mtype, env, hp, agent, alldyn, sessions, useweight=None, nocue=None, noreward=None):
    lat = np.zeros(sessions)
    mvpath = np.zeros((3,6,env.normax,2))
    dgr = []
    env.make(mtype=mtype, nocue=nocue, noreward=noreward)

    if useweight:
        agent.model.set_weights(useweight)

    for t in range(sessions*6):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.ac.reset()
        agent.cri_reset()
        wtrack = []
        value = agent.vstate
        rfr = agent.rstate
        rho = agent.ac.qstate

        if t%6==0:
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

            # Pass coordinates to Place Cell & LCM to get actor
            rfr, q, _ = agent.act(state=state, cue_r_fb=cue)

            # Convolve actor dynamics & Action selection
            action, rho = agent.ac.move(q)

            # Use action on environment, ds4r: distance from reward
            state, cue, reward, done, ds4r = env.step(action)

            # plasticity using Backward euler
            if hp['eulerm'] == 0:
                rpe, value = agent.learn(pre=rfr, post=rho, R=reward, v=value, plasticity=plastic, s1=state,
                                         cue_r1_fb=cue)

            # save lsm & actor dynamics for analysis
            if t in env.nort:
                save_rdyn(alldyn[0], mtype, t, env.startpos, env.cue, rfr)
                save_rdyn(alldyn[1], mtype, t, env.startpos, env.cue, rho)
                save_rdyn(alldyn[2], mtype, t, env.startpos, env.cue, value)
                save_rdyn(alldyn[3], mtype, t, env.startpos, env.cue, agent.tderr)
            else:
                wtrack.append([agent.dwc,agent.dwa])

            if done:
                break

        if t in env.nort:
            sesslat.append(np.nan)
            dgr.append(env.dgr)
            sid = np.argmax(np.array(noreward) == (t // 6) + 1)
            mvpath[sid, env.idx] = np.array(env.tracks)[:env.normax]
        else:
            sesslat.append(env.i)
            alldyn[4].append(np.sum(np.array(wtrack), axis=0))

        if (t + 1) % 6 == 0:
            lat[((t + 1) // 6) - 1] = np.mean(sesslat)

        if hp['platform'] == 'laptop' or b == 0:
            # Trial information
            print('T {} | C {} | S {} | TD {:4.3f} | D {:4.3f} | st {} | Dgr {}'.format(
                t, find_cue(env.cue), env.i // (1000 // env.tstep), rpe.numpy()[0][0], ds4r, env.startpos[0], np.round(dgr,1)))

            # Session information
            if (t + 1) % 6 == 0:
                print('################## {} Session {}/{}, Avg Steps {:5.1f}, PI {} ################'.format(
                    mtype, (t + 1) // 6, sessions, lat[((t + 1) // 6) - 1], env.sessr))

    # get mean visit rate
    sesspi = np.array(dgr) > pithres
    sesspi = np.sum(np.array(sesspi).reshape(len(noreward), 6), axis=1)
    dgr = np.mean(np.array(dgr).reshape(len(noreward), 6), axis=1)
    mdlw = agent.model.get_weights()
    if hp['platform'] == 'server':
        print('Agent {} {} training dig rate: {}'.format(b, mtype, dgr))
    return lat, mvpath, mdlw, dgr, sesspi


def main_multiplepa_expt(hp,b):
    import tensorflow as tf
    from backend_scripts.model import SimpleAgent

    print('Agent {} started training ...'.format(b))
    exptname = hp['exptname']
    print(exptname)

    # create environment
    env = Maze(hp)
    agent = SimpleAgent(hp=hp, env=env)
    trsess = hp['trsess']

    # Create nonrewarded probe trial index
    scl = trsess // 20  # scale number of sessions to Tse et al., 2007
    nonrp = [2 * scl, 9 * scl, 16 * scl]  # sessions that are non-rewarded probe trials

    # Start experiment
    rdyn = {}
    qdyn = {}
    cdyn = {}
    tdyn = {}
    wtrk = []
    alldyn = [rdyn,qdyn,cdyn,tdyn, wtrk]
    tf.compat.v1.reset_default_graph()
    start = dt.time()

    # Start Training
    lat, mvpath, trw, dgr, pi = run_multiple_expt(b, 'train',env,hp,agent,alldyn, trsess,noreward=nonrp)

    if hp['savevar']:
        saveload('save', './Data/vars_{}_{}'.format(exptname, dt.time()),
                 [rdyn, qdyn, cdyn, tdyn, wtrk, mvpath, lat, dgr, pi, trw])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, pi, mvpath, trw, alldyn


''' Reservoir '''
def run_res_multiple_expt(b, mtype, env, hp, agent, alldyn, sessions, useweight=None, nocue=None, noreward=None):
    lat = np.zeros(sessions)
    mvpath = np.zeros((3,6,env.normax,2))
    dgr = []
    env.make(mtype=mtype, nocue=nocue, noreward=noreward)

    if useweight:
        agent.model.set_weights(useweight)

    for t in range(sessions*6):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.ac.reset()
        agent.cri_reset()
        rstate = agent.rstate
        rfr = agent.rstate
        value = agent.vstate
        rho = agent.ac.qstate
        wtrack = []

        if t%6==0:
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
                rpe, value = agent.learn(s1=state, cue_r1_fb=cue,plasticity=plastic,
                                  pre=rfr, post=rho, R=reward, v=value, h1=rstate)

            # Pass coordinates to Place Cell & LCM to get actor & critic values
            q, rfr, rstate, _ = agent.act(state=state, cue_r_fb=cue, rstate=rstate)

            # Convolve actor dynamics & Action selection
            action, rho = agent.ac.move(q)

            # Use action on environment, ds4r: distance from reward
            state, cue, reward, done, ds4r = env.step(action)

            # plasticity using Backward euler
            if hp['eulerm'] == 0:
                rpe, value = agent.learn(s1=state, cue_r1_fb=cue,plasticity=plastic,
                                  pre=rfr, post=rho, R=reward, v=value, h1=rstate)
            # save lsm & actor dynamics for analysis
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
            sesslat.append(np.nan)
            dgr.append(env.dgr)
            sid = np.argmax(np.array(noreward) == (t // 6) + 1)
            mvpath[sid, env.idx] = np.array(env.tracks)[:env.normax]

        else:
            sesslat.append(env.i)
            alldyn[4].append(np.sum(np.array(wtrack), axis=0))

        if (t + 1) % 6 == 0:
            lat[((t + 1) // 6) - 1] = np.mean(sesslat)

        if hp['platform'] == 'laptop' or b==0:
            # Trial information
            print('T {} | C {} | S {} | TD {:4.3f} | D {:4.3f} | st {} | Dgr {}'.format(
                t, find_cue(env.cue), env.i // (1000 // env.tstep), rpe.numpy()[0][0], ds4r, env.startpos[0], np.round(dgr,1)))

            # Session information
            if (t + 1) % 6 == 0:
                print('################## {} Session {}/{}, Avg Steps {:5.1f}, PI {} ################'.format(
                    mtype, (t + 1) // 6, sessions, lat[((t + 1) // 6) - 1], env.sessr))

    # get mean visit rate
    sesspi = np.array(dgr) > pithres
    sesspi = np.sum(np.array(sesspi).reshape(len(noreward), 6), axis=1)
    dgr = np.mean(np.array(dgr).reshape(len(noreward), 6), axis=1)

    mdlw = agent.model.get_weights()

    if hp['platform'] == 'server':
        print('Agent {} {} training dig rate: {}'.format(b, mtype, dgr))
    return lat, mvpath, mdlw, dgr, sesspi


def main_res_multiplepa_expt(hp,b):
    import tensorflow as tf
    from backend_scripts.model import LSMAgent

    print('Agent {} started training ...'.format(b))
    exptname = hp['exptname']
    print(exptname)

    # create environment
    env = Maze(hp)
    trsess = hp['trsess']

    # Create nonrewarded probe trial index
    scl = trsess // 20  # scale number of sessions to Tse et al., 2007
    nonrp = [2 * scl, 9 * scl, 16 * scl]  # sessions that are non-rewarded probe trials

    # Start experiment
    rdyn = {}
    qdyn = {}
    cdyn = {}
    tdyn = {}
    wtrk = []
    alldyn = [rdyn,qdyn,cdyn,tdyn, wtrk]
    tf.compat.v1.reset_default_graph()
    start = dt.time()
    agent = LSMAgent(hp=hp,env=env)

    # Start Training
    lat, mvpath, trw, dgr, pi = run_res_multiple_expt(b, 'train',env,hp,agent,alldyn, trsess,noreward=nonrp)

    if hp['savevar']:
        saveload('save', './Data/vars_{}_{}'.format(exptname, dt.time()),
                 [rdyn, qdyn, cdyn, tdyn, wtrk, mvpath, lat, dgr, pi, trw])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, pi, mvpath, trw, alldyn

''' Working memory tasks with bump attractor '''

''' Feedforwad models'''

def run_wkm_multiple_expt(b,mtype, env, hp, agent, alldyn, sessions, useweight=None, nocue=None, noreward=None):
    lat = np.zeros(sessions)
    mvpath = np.zeros((3,6,env.normax,2))
    dgr = []
    env.make(mtype=mtype, nocue=nocue, noreward=noreward)

    if useweight:
        agent.model.set_weights(useweight)

    for t in range(sessions*6):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.ac.reset()
        agent.cri_reset()
        wtrack = []
        value = agent.vstate
        rfr = agent.rstate
        rho = agent.ac.qstate
        mstate, mem = np.random.normal(size=[1, agent.nwm], scale=0.1), np.zeros([1, agent.nwm])

        if t%6==0:
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
                rpe, value = agent.learn(pre=rfr, post=rho, R=reward, v=value,plasticity=plastic,s1=state,
                                         cue_r1_fb=np.concatenate([mem[0],cue]),m1=mstate)

            # Pass coordinates to Place Cell & LCM to get actor
            rfr, q, _, mem, mstate = agent.act(state=state, cue_r_fb=np.concatenate([mem[0],cue]), mstate=mstate)

            # Convolve actor dynamics & Action selection
            action, rho = agent.ac.move(q)

            # Use action on environment, ds4r: distance from reward
            state, cue, reward, done, ds4r = env.step(action)

            # plasticity using Backward euler
            if hp['eulerm'] == 0:
                rpe, value = agent.learn(pre=rfr, post=rho, R=reward, v=value, plasticity=plastic, s1=state,
                                         cue_r1_fb=np.concatenate([mem[0],cue]),m1=mstate)

            # save lsm & actor dynamics for analysis
            if t in env.nort:
                save_rdyn(alldyn[0], mtype, t, env.startpos, env.cue, rfr)
                save_rdyn(alldyn[1], mtype, t, env.startpos, env.cue, rho)
                save_rdyn(alldyn[2], mtype, t, env.startpos, env.cue, value)
                save_rdyn(alldyn[3], mtype, t, env.startpos, env.cue, agent.tderr)
            else:
                wtrack.append([agent.dwc,agent.dwa])

            if done:
                break

        if t in env.nort:
            sesslat.append(np.nan)
            dgr.append(env.dgr)
            sid = np.argmax(np.array(noreward) == (t // 6) + 1)
            mvpath[sid, env.idx] = np.array(env.tracks)[:env.normax]
        else:
            sesslat.append(env.i)
            alldyn[4].append(np.sum(np.array(wtrack), axis=0))

        if (t + 1) % 6 == 0:
            lat[((t + 1) // 6) - 1] = np.mean(sesslat)

        if hp['platform'] == 'laptop' or b == 0:
            # Trial information
            print('T {} | C {} | S {} | TD {:4.3f} | D {:4.3f} | st {} | Dgr {}'.format(
                t, find_cue(env.cue), env.i // (1000 // env.tstep), rpe.numpy()[0][0], ds4r, env.startpos[0], np.round(dgr,1)))

            # Session information
            if (t + 1) % 6 == 0:
                print('################## {} Session {}/{}, Avg Steps {:5.1f}, PI {} ################'.format(
                    mtype, (t + 1) // 6, sessions, lat[((t + 1) // 6) - 1], env.sessr))

    # get mean visit rate
    sesspi = np.array(dgr) > pithres
    sesspi = np.sum(np.array(sesspi).reshape(len(noreward), 6), axis=1)
    dgr = np.mean(np.array(dgr).reshape(len(noreward), 6), axis=1)
    mdlw = agent.model.get_weights()
    if hp['platform'] == 'server':
        print('Agent {} {} training dig rate: {}'.format(b, mtype, dgr))
    return lat, mvpath, mdlw, dgr, sesspi


def wkm_multiplepa_expt(hp,b):
    import tensorflow as tf
    from backend_scripts.model import BumpSimpleAgent

    print('Agent {} started training ...'.format(b))
    exptname = hp['exptname']
    print(exptname)

    # create environment
    env = Maze(hp)
    agent = BumpSimpleAgent(hp=hp, env=env)
    trsess = hp['trsess']

    # Create nonrewarded probe trial index
    scl = trsess // 20  # scale number of sessions to Tse et al., 2007
    nonrp = [2 * scl, 9 * scl, 16 * scl]  # sessions that are non-rewarded probe trials

    # Start experiment
    rdyn = {}
    qdyn = {}
    cdyn = {}
    tdyn = {}
    wtrk = []
    alldyn = [rdyn,qdyn,cdyn,tdyn, wtrk]
    tf.compat.v1.reset_default_graph()
    start = dt.time()

    # Start Training
    lat, mvpath, trw, dgr, pi = run_wkm_multiple_expt(b, 'train',env,hp,agent,alldyn, trsess,noreward=nonrp)

    if hp['savevar']:
        saveload('save', './Data/vars_{}_{}'.format(exptname, dt.time()),
                 [rdyn, qdyn, cdyn, tdyn, wtrk, mvpath, lat, dgr, pi, trw])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, pi, mvpath, trw, alldyn


''' Reservoir '''
def run_wkm_res_multiple_expt(b, mtype, env, hp, agent, alldyn, sessions, useweight=None, nocue=None, noreward=None):
    lat = np.zeros(sessions)
    mvpath = np.zeros((3,6,env.normax,2))
    dgr = []
    env.make(mtype=mtype, nocue=nocue, noreward=noreward)

    if useweight:
        agent.model.set_weights(useweight)

    for t in range(sessions*6):
        # Reset environment, actor dynamics
        state, cue, reward, done = env.reset(trial=t)
        agent.ac.reset()
        agent.cri_reset()
        wtrack = []
        value = agent.vstate
        rfr = agent.rstate
        rstate = agent.rstate
        rho = agent.ac.qstate
        mstate, mem = np.random.normal(size=[1, agent.nwm], scale=0.1), np.zeros([1, agent.nwm])

        if t%6==0:
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
                rpe, value = agent.learn(s1=state, cue_r1_fb=np.concatenate([mem[0],cue]),plasticity=plastic,
                                  pre=rfr, post=rho, R=reward, v=value, h1=[rstate,mstate])

            # Pass coordinates to Place Cell & LCM to get actor & critic values
            q, rfr, rstate, _, mem, mstate = agent.act(state=state, cue_r_fb=np.concatenate([mem[0],cue]), rstate=[rstate, mstate])

            # Convolve actor dynamics & Action selection
            action, rho = agent.ac.move(q)

            # Use action on environment, ds4r: distance from reward
            state, cue, reward, done, ds4r = env.step(action)

            # plasticity using Backward euler
            if hp['eulerm'] == 0:
                rpe, value = agent.learn(s1=state, cue_r1_fb=np.concatenate([mem[0],cue]),plasticity=plastic,
                                  pre=rfr, post=rho, R=reward, v=value, h1=[rstate,mstate])

            # save lsm & actor dynamics for analysis
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
            sesslat.append(np.nan)
            dgr.append(env.dgr)
            sid = np.argmax(np.array(noreward) == (t // 6) + 1)
            mvpath[sid, env.idx] = np.array(env.tracks)[:env.normax]
        else:
            sesslat.append(env.i)
            alldyn[4].append(np.sum(np.array(wtrack), axis=0))

        if (t + 1) % 6 == 0:
            lat[((t + 1) // 6) - 1] = np.mean(sesslat)

        if hp['platform'] == 'laptop' or b == 0:
            # Trial information
            print('T {} | C {} | S {} | TD {:4.3f} | D {:4.3f} | st {} | Dgr {}'.format(
                t, find_cue(env.cue), env.i // (1000 // env.tstep), rpe.numpy()[0][0], ds4r, env.startpos[0], np.round(dgr,1)))

            # Session information
            if (t + 1) % 6 == 0:
                print('################## {} Session {}/{}, Avg Steps {:5.1f}, PI {} ################'.format(
                    mtype, (t + 1) // 6, sessions, lat[((t + 1) // 6) - 1], env.sessr))

    # get mean visit rate
    sesspi = np.array(dgr) > pithres
    sesspi = np.sum(np.array(sesspi).reshape(len(noreward), 6), axis=1)
    dgr = np.mean(np.array(dgr).reshape(len(noreward), 6), axis=1)
    mdlw = agent.model.get_weights()
    if hp['platform'] == 'server':
        print('Agent {} {} training dig rate: {}'.format(b, mtype, dgr))
    return lat, mvpath, mdlw, dgr, sesspi


def wkm_res_multiplepa_expt(hp,b):
    import tensorflow as tf
    from backend_scripts.model import BumpLSMAgent

    print('Agent {} started training ...'.format(b))
    exptname = hp['exptname']
    print(exptname)

    # create environment
    env = Maze(hp)
    trsess = hp['trsess']

    # Create nonrewarded probe trial index
    scl = trsess // 20  # scale number of sessions to Tse et al., 2007
    nonrp = [2 * scl, 9 * scl, 16 * scl]  # sessions that are non-rewarded probe trials

    # Start experiment
    rdyn = {}
    qdyn = {}
    cdyn = {}
    tdyn = {}
    wtrk = []
    alldyn = [rdyn,qdyn,cdyn,tdyn, wtrk]
    tf.compat.v1.reset_default_graph()
    start = dt.time()
    agent = BumpLSMAgent(hp=hp,env=env)

    # Start Training
    lat, mvpath, trw, dgr, pi = run_wkm_res_multiple_expt(b, 'train',env,hp,agent,alldyn, trsess,noreward=nonrp)

    if hp['savevar']:
        saveload('save', './Data/vars_{}_{}'.format(exptname, dt.time()),
                 [rdyn, qdyn, cdyn, tdyn, wtrk, mvpath, lat, dgr, pi, trw])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, pi, mvpath, trw, alldyn