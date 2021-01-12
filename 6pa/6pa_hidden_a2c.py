import numpy as np
import matplotlib.pyplot as plt
from backend_scripts.utils import save_rdyn, find_cue, saveload, plot_dgr, get_default_hp, plot_maps
import time as dt
from backend_scripts.maze_env import Maze
import tensorflow as tf
from backend_scripts.model import choose_activation, place_cells, action_cells
import multiprocessing as mp
from functools import partial

class BackpropAgent:
    def __init__(self, hp, env):
        ''' environment parameters '''
        self.env = env
        self.tstep = hp['tstep']

        ''' agent parameters '''
        self.taug = hp['taug']
        self.beg = (1 - (self.tstep / self.taug))  # taug for euler backward approximation
        self.lr = hp['lr']
        self.npc = hp['npc']
        self.nact = hp['nact']
        self.workmem = hp['workmem']
        self.rstate = tf.zeros([1,hp['nhid']])
        self.action = np.zeros(2)
        self.actalpha = hp['actalpha']

        ''' critic parameters '''
        self.ncri = hp['ncri']
        self.vstate = tf.zeros([1, self.ncri])
        self.vscale = hp['vscale']
        self.calpha = hp['tstep']/hp['ctau']
        self.criact = choose_activation(hp['criact'],hp)
        self.eulerm = hp['eulerm']
        self.maxcritic = 0
        self.loss = 0

        ''' Setup model: Place cell --> Action cells '''
        self.pc = place_cells(hp)
        self.model = BackpropModel(hp)
        self.ac = action_cells(hp)
        self.memory = Memory()
        self.opt = tf.optimizers.RMSprop(learning_rate=self.lr)
        self.eb = hp['entbeta']
        self.va = hp['valalpha']

    def act(self, state, cue_r_fb):
        s = self.pc.sense(state)  # convert coordinate info to place cell activity
        state_cue_fb = np.concatenate([s, cue_r_fb])  # combine all inputs

        if self.workmem and self.env.i <= self.env.workmemt:
            # silence state presentation during cue presentation
            state_cue_fb[:self.npc ** 2] = 0

        if self.env.done:
            # silence all inputs after trial ends
            state_cue_fb = np.zeros_like(state_cue_fb)

        ''' Predict next action '''
        r, q, c = self.model(tf.cast(state_cue_fb[None, :], dtype=tf.float32))

        # stochastic discrete action selection
        action_prob_dist = tf.nn.softmax(q)
        actsel = np.random.choice(range(self.nact), p=action_prob_dist.numpy()[0])
        actdir = self.ac.aj[:,actsel]/self.tstep # constant speed 0.03
        self.action = (1-self.actalpha)*self.action + self.actalpha*actdir.numpy()

        return state_cue_fb, r, q, c, actsel, self.action

    def replay(self):
        discount_reward = self.discount_normalise_rewards(self.memory.rewards)

        with tf.GradientTape() as tape:
            policy_loss, value_loss, total_loss = self.compute_loss(self.memory, discount_reward)

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.model.trainable_weights))
        self.tderr = tf.reshape(total_loss, (1, 1))

        return policy_loss, value_loss, total_loss

    def discount_normalise_rewards(self, rewards):
        discounted_rewards = []
        cumulative = 0
        for reward in rewards[::-1]:
            cumulative = reward + self.beg * cumulative
            discounted_rewards.append(cumulative)
        discounted_rewards.reverse()

        return discounted_rewards

    def compute_loss(self, memory, discounted_rewards):
        _, logit, values = self.model(tf.convert_to_tensor(np.vstack(memory.states), dtype=tf.float32))

        # Advantage = Discounted R - V(s) = TD error
        advantage = tf.convert_to_tensor(np.array(discounted_rewards), dtype=tf.float32) - values[:,0]

        value_loss = advantage**2

        # compute actor policy loss
        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=np.array(memory.actions))
        policy_loss = neg_log_prob * tf.stop_gradient(advantage)

        # compute entropy & add negative to prevent faster convergence of actions & better initial exploration
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=tf.nn.softmax(logit))

        # merge all losses to train network tgt
        comb_loss = tf.reduce_mean((self.va * value_loss + policy_loss + self.eb * entropy))
        self.loss = comb_loss

        return policy_loss, value_loss, comb_loss

    def cri_reset(self):
        self.vstate = tf.zeros([1, self.ncri])


class BackpropModel(tf.keras.Model):
    def __init__(self, hp):
        super(BackpropModel, self).__init__()
        self.nact = hp['nact']
        self.ncri = hp['ncri']
        self.nhid = hp['nhid']
        self.npc = hp['npc']
        self.hidscale = hp['hidscale']
        self.crins = np.sqrt(hp['ctau']/hp['tstep']) * hp['crins']
        self.actns = np.sqrt(hp['ctau'] / hp['tstep']) * hp['actns']
        self.hidact = hp['hidact']
        self.hidscale = hp['hidscale']

        if hp['controltype'] == 'expand':
            self.controltype = (self.nhid // (self.npc ** 2 + hp['cuesize'])) + 1  # tile factor 16 (1024) or 80 (8192)

        elif hp['controltype'] == 'hidden':
            self.controltype = tf.keras.layers.Dense(units=self.nhid,
                                                     activation=choose_activation(self.hidact, hp),
                                                     use_bias=False, name='hidden',
                                                     kernel_initializer=
                                                     tf.keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None))
        else:
            self.controltype = choose_activation(self.hidact,hp)

        self.critic = tf.keras.layers.Dense(units=self.ncri, activation='linear',
                                            use_bias=False, kernel_initializer='zeros', name='critic')
        self.actor = tf.keras.layers.Dense(units=self.nact, activation='linear',
                                           use_bias=False, kernel_initializer='zeros', name='actor')

    def call(self, inputs):
        if isinstance(self.controltype, int):
            r = self.hidscale * tf.tile(inputs, [1, self.controltype])
        else:
            r = self.hidscale * self.controltype(inputs)
        c = self.critic(r)
        q = self.actor(r)
        return r, q, c


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

pithres = 40

def multiplepa_script(hp):
    exptname = hp['exptname']
    btstp = hp['btstp']

    print(exptname)

    # store performance
    latflag = False
    totlat = np.zeros([btstp, hp['trsess']])
    totdgr = np.zeros([btstp, 3])
    totpi = np.zeros_like(totdgr)
    scl = hp['trsess'] // 20  # scale number of sessions to Tse et al., 2007

    pool = mp.Pool(processes=hp['cpucount'])

    x = pool.map(partial(main_multiplepa_expt, hp), np.arange(btstp))

    # Start experiment
    for b in range(btstp):
        totlat[b], totdgr[b], totpi[b], trw, mvpath, alldyn =  x[b]

    if latflag:
        allatency = np.mean(totlat,axis=2)
        firstlatency = np.mean(totlat[:,:,:,0],axis=2)
    else:
        firstlatency = allatency = totlat

    plt.figure(figsize=(15, 8))
    plt.gcf().text(0.01, 0.01, exptname, fontsize=10)
    plt.subplot(231)
    plt.title('Latency')
    plt.errorbar(x=np.arange(firstlatency.shape[1]), y =np.mean(firstlatency, axis=0), yerr=np.std(firstlatency,axis=0))
    plt.plot(np.mean(allatency,axis=0),linewidth=3)

    plot_dgr(totdgr, scl, 232, 6)

    plot_maps(alldyn, mvpath, hp, 233)

    env = Maze(hp)
    env.make('train')

    col = ['b', 'g', 'r', 'y', 'm', 'k']
    for i, k in zip(np.arange(4, 7), [mvpath[0], mvpath[1], mvpath[2]]):
        plt.subplot(2, 3, i)
        plt.title('PS{}'.format(i-3))
        for pt in range(len(mvpath[2])):
            plt.plot(np.array(k[pt])[:, 0], np.array(k[pt])[:, 1], col[pt], alpha=0.5)
            circle = plt.Circle(env.rlocs[pt], env.rrad, color=col[pt])
            plt.gcf().gca().add_artist(circle)
        plt.axis((-env.au / 2, env.au / 2, -env.au / 2, env.au / 2))

    print(exptname)

    plt.tight_layout()

    if hp['task'] == '6pa':
        if hp['savefig']:
            plt.savefig('./6pa/Fig/fig_{}.png'.format(exptname))
        if hp['savegenvar']:
            saveload('save', './6pa/Data/genvars_{}_b{}_{}'.format(exptname, btstp, dt.time()),
                     [totlat, totdgr, totpi])
    else:
        if hp['savefig']:
            plt.savefig('./wkm/Fig/fig_{}.png'.format(exptname))
        if hp['savegenvar']:
            saveload('save', './wkm/Data/genvars_{}_b{}_{}'.format(exptname,btstp, dt.time()),
                     [totlat, totdgr, totpi])

    return totlat, totdgr, totpi, trw, mvpath, alldyn

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
        agent.memory.clear()

        if t%6==0:
            sesslat = []

        while not done:
            if env.rendercall:
                env.render()

            # Pass coordinates to Place Cell & LCM to get actor & critic values
            allstate, rfr, rho, value, actsel, action = agent.act(state=state, cue_r_fb=cue)

            # Use action on environment, ds4r: distance from reward
            state, cue, reward, done, ds4r = env.step(action)

            if reward <= 0 and done:
                reward = -1
            elif reward > 0:
                reward = 1
                done = True

            agent.memory.store(state=allstate, action=actsel,reward=reward)

            # save lsm & actor dynamics for analysis
            if t in env.nort:
                save_rdyn(alldyn[0], mtype, t, env.startpos, env.cue, rfr)
                save_rdyn(alldyn[1], mtype, t, env.startpos, env.cue, rho)
                save_rdyn(alldyn[2], mtype, t, env.startpos, env.cue, value)
                save_rdyn(alldyn[3], mtype, t, env.startpos, env.cue, agent.tderr)

            if done:
                if t not in env.nort:
                    agent.replay()
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
            print('T {} | C {} | S {} | TD {:4.3f} | D {:4.3f} | st {} | Dgr {} | mlen {}'.format(
                t, find_cue(env.cue), env.i // (1000 // env.tstep), agent.loss, ds4r, env.startpos[0], np.round(dgr,1), len(agent.memory.rewards)))

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
    env = Maze(hp)
    agent = BackpropAgent(hp=hp, env=env)
    print('Agent {} started training ...'.format(b))
    exptname = hp['exptname']
    print(exptname)

    # create environment
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
        saveload('save', './6pa/Data/vars_{}_{}'.format(exptname, dt.time()),
                 [rdyn, qdyn, cdyn, tdyn, wtrk, mvpath, lat, dgr, pi, trw])

    print('---------------- Agent {} done in {:3.2f} min ---------------'.format(b, (dt.time() - start) / 60))

    return lat, dgr, pi, trw, mvpath, alldyn


if __name__ == '__main__':

    hp = get_default_hp(task='6pa',platform='server')

    hp['controltype'] = 'hidden'  # expand, hidden, classic
    hp['tstep'] = 100  # deltat = 100ms ** A2C algorithm tested only at dt = 100ms
    hp['trsess'] = 100
    hp['btstp'] = 10
    hp['time'] = 600  # Tmax seconds
    hp['savefig'] = True
    hp['savevar'] = False
    hp['saveweight'] = False
    hp['savegenvar'] = False

    ''' Hidden parameters '''
    hp['nhid'] = 8192  # number of hidden units
    hp['hidact'] = 'relu'
    hp['K'] = None
    hp['sparsity'] = 0

    ''' Other Model parameters '''
    hp['lr'] = 0.00001
    hp['taug'] = 10000
    hp['actalpha'] = 1/4
    hp['maxspeed'] = 0.07  # max step size per 100ms

    hp['entbeta'] = -0.001  # 0.0001
    hp['valalpha'] = 0.5

    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = 'A2C_{}tg_{}av_{}be_{}_{}_{}n_{}ha_{}lr_{}dt_b{}_{}'.format(hp['taug'],hp['valalpha'],hp['entbeta'],
        hp['task'], hp['controltype'],hp['nhid'], hp['hidact'],hp['lr'], hp['tstep'],hp['btstp'],dt.monotonic())

    totlat, totdgr, totpi, trw, mvpath, alldyn = multiplepa_script(hp)