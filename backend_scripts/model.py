import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

'''
Simple Agents
'''


class SimpleAgent:
    def __init__(self, hp, env):
        ''' environment parameters '''
        self.env = env
        self.tstep = hp['tstep']

        ''' agent parameters '''
        self.taug = hp['taug']
        self.beg = (1 - (self.tstep / self.taug))  # taur for backward euler continuous TD
        self.feg = (1 + (self.tstep / self.taug))  # taur for forward euler continuous TD
        self.lr = hp['lr']
        self.npc = hp['npc']
        self.nact = hp['nact']
        self.workmem = hp['workmem']
        if hp['controltype'] == 'expand':
            self.rstate = tf.zeros([1, 67*((hp['nhid'] // (self.npc ** 2 + hp['cuesize'])) + 1)])
        else:
            self.rstate = tf.zeros([1, hp['nhid']])

        ''' critic parameters '''
        self.ncri = hp['ncri']
        self.vstate = tf.zeros([1, self.ncri])
        self.vscale = hp['vscale']
        self.calpha = hp['tstep']/hp['ctau']
        self.criact = choose_activation(hp['criact'],hp)
        self.eulerm = hp['eulerm']

        ''' Setup model: Place cell --> Action cells '''
        self.pc = place_cells(hp)
        self.model = SimpleModel(hp)
        self.ac = action_cells(hp)

    def act(self, state, cue_r_fb):
        s = self.pc.sense(state)  # convert coordinate info to place cell activity
        state_cue_fb = np.concatenate([s, cue_r_fb])  # combine all inputs

        if self.workmem and self.env.i <= self.env.workmemt:
            # silence state presentation during cue presentation
            state_cue_fb[:self.npc ** 2] = 0

        ''' Predict next action '''
        x, q, c = self.model(tf.cast(state_cue_fb[None, :], dtype=tf.float32))

        return x, q, c

    def learn(self, s1, cue_r1_fb, pre, post, R, v, plasticity=True):
        ''' Hebbian rule: lr * TD * eligibility trace '''

        _, _, c2 = self.act(s1, cue_r1_fb)
        self.vstate = (1 - self.calpha) * self.vstate + self.calpha * c2
        v2 = self.criact(self.vstate)

        if self.eulerm == 0:
            # backward euler method
            tderr = R + tf.reshape(tf.reduce_mean(self.beg * v2 - v),[1,1]) / self.tstep
        elif self.eulerm == 1:
            # forward euler method
            tderr = R + tf.reshape(tf.reduce_mean(v2 - self.feg * v), [1, 1]) / self.tstep
        self.tderr = tf.reshape(tderr,(1,1))

        if plasticity:
            dwc = self.vscale * self.tstep * self.lr * tderr * tf.transpose(pre)
            self.model.layers[-2].set_weights([self.model.layers[-2].get_weights()[0] + dwc])

            ea = tf.linalg.matmul(pre, tf.cast(post, dtype=tf.float32), transpose_a=True)
            dwa = self.tstep * self.lr * tderr * ea
            self.model.layers[-1].set_weights([self.model.layers[-1].get_weights()[0] + dwa])

            self.dwa = np.sum(abs(dwa))
            self.dwc = np.sum(abs(dwc))
        return tderr, v2

    def cri_reset(self):
        self.vstate = tf.zeros([1, self.ncri])


class SimpleModel(tf.keras.Model):
    def __init__(self, hp):
        super(SimpleModel, self).__init__()
        self.nact = hp['nact']
        self.ncri = hp['ncri']
        self.nhid = hp['nhid']
        self.npc = hp['npc']
        self.hidscale = hp['hidscale']
        self.crins = np.sqrt(hp['ctau']/hp['tstep']) * hp['crins']
        self.hidact = hp['hidact']
        self.hidscale = hp['hidscale']

        if hp['controltype'] == 'expand':
            self.controltype = (self.nhid // (self.npc ** 2 + hp['cuesize'])) + 1  # tile factor 16 (1024) or 80 (8192)

        elif hp['controltype'] == 'hidden':
            if hp['K'] is None:
                hidwin = tf.keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None)
            else:
                hidwin = tf.constant_initializer(set_excitatory(matsize=(67,self.nhid),K=hp['K']))
            self.controltype = tf.keras.layers.Dense(units=self.nhid,
                                                     activation=choose_activation(self.hidact, hp),
                                                     use_bias=False, name='hidden',
                                                     kernel_initializer=hidwin)
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
        c = self.critic(r) + tf.random.normal(shape=(1,self.ncri),stddev=self.crins)
        q = self.actor(r)
        return r, q, c


class RNNCell(tf.keras.layers.Layer):
    def __init__(self,hp, ninput):
        super(RNNCell, self).__init__()
        self.nrnn = hp['nhid']
        self.ralpha = hp['tstep'] / hp['rtau']
        self.recns = np.sqrt(1 / self.ralpha) * hp['resns']
        self.resact = choose_activation(hp['hidact'],hp)
        self.resrecact = choose_activation(hp['resrecact'],hp)
        self.cp = hp['cp']
        self.recwinscl = hp['recwinscl']

        ''' win weight init'''
        winconn = np.random.uniform(-self.recwinscl, self.recwinscl, (ninput, self.nrnn ))  # uniform dist [-1,1]
        winprob = np.random.choice([0, 1], (ninput, self.nrnn ), p=[1 - self.cp[0], self.cp[0]])
        w_in = np.multiply(winconn, winprob)

        ''' wrec weight init '''
        connex = np.random.normal(0, np.sqrt(1 / (self.cp[1] * self.nrnn)), size=(self.nrnn, self.nrnn))
        prob = np.random.choice([0, 1], (self.nrnn , self.nrnn ), p=[1 - self.cp[1], self.cp[1]])
        w_rec = np.multiply(connex, prob)  # initialise random network with connection probability
        # w_rec *= (np.eye(nrnn) == 0)  # remove self recurrence
        self.wrnn = np.concatenate((w_in, w_rec * hp['chaos']), axis=0)  # add gain parameter to recurrent weights

        # w,v = np.linalg.eig(w_rec* lambda_chaos)
        # spectralradius = np.max(abs(w)) # compute spectral radius

    @property
    def state_size(self):
        return self.nrnn

    @property
    def output_size(self):
        return self.nrnn

    def build(self, inputs_shape):
        if inputs_shape[1] is None:
            raise ValueError("Expected inputs.shape[-1] but shape: %s" % inputs_shape)

        self.rkernel = self.add_weight(
            'W_in_W_rec',
            shape=[inputs_shape[1] + self.nrnn, self.nrnn],
            initializer=tf.constant_initializer(self.wrnn))

        self.built = True

    def call(self, inputs, state):
        """ Reservoir without bias: membrane potential = old + act(Win * input +  Wrec * state + Wfb * z)"""

        resinput = tf.matmul(tf.concat([inputs, self.resrecact(state[0])], 1), self.rkernel)

        xt = (1 - self.ralpha) * state[0] + self.ralpha * (resinput + tf.random.normal(shape=(1, self.nrnn), mean=0,
                                                                                          stddev=self.recns))

        rt = self.resact(xt)
        return rt, xt


class LSMAgent:
    def __init__(self, hp, env):
        ''' Copy environment parameters '''
        self.env = env
        self.tstep = hp['tstep']

        ''' agent parameters '''
        self.taug = hp['taug']
        self.beg = (1 - (self.tstep / self.taug))  # taur for backward euler continuous TD
        self.feg = (1 + (self.tstep / self.taug))  # taur for forward euler continuous TD
        self.lr = hp['lr']
        self.npc = hp['npc']
        self.nact = hp['nact']
        self.workmem = hp['workmem']
        self.nrnn = hp['nhid']
        self.actns = hp['actns']
        self.qalpha = hp['tstep']/hp['qtau']
        self.rstate = tf.zeros([1, hp['nhid']])

        ''' critic parameters '''
        self.ncri = hp['ncri']
        self.vstate = tf.zeros([1, self.ncri])
        self.vscale = hp['vscale']
        self.calpha = hp['tstep']/hp['ctau']
        self.criact = choose_activation(hp['criact'],hp)
        self.eulerm = hp['eulerm']
        self.fbsz = hp['fbsz']

        ''' Setup model: Place cell --> Liquid Computing Model --> Action cells '''
        self.pc = place_cells(hp)
        self.model = LSMmodel(hp)
        self.ac = action_cells(hp)

    def act(self, state, cue_r_fb, rstate):
        s = self.pc.sense(state)  # convert coordinate info to place cell activity
        state_cue_fb = np.concatenate([s, cue_r_fb])  # combine all inputs

        if self.workmem and self.env.i <= self.env.workmemt:
            # silence state presentation during cue presentation
            state_cue_fb[:self.npc ** 2] = 0

        ''' Model prediction '''
        rfr, h, q, c = self.model(tf.cast(state_cue_fb[None, None, :], dtype=tf.float32), rstate)

        return q, rfr, h, c

    def learn(self, s1, cue_r1_fb, h1, pre, post, R, v,plasticity=True):
        ''' Hebbian rule: lr * TD * eligibility trace '''

        _, _, _, c2 = self.act(s1, cue_r1_fb, h1)
        self.vstate = (1 - self.calpha) * self.vstate + self.calpha * c2
        v2 = self.criact(self.vstate)

        if self.eulerm == 0:
            # backward euler method
            tderr = R + tf.reshape(tf.reduce_mean(self.beg * v2 - v),[1,1]) / self.tstep
        elif self.eulerm == 1:
            # forward euler method
            tderr = R + tf.reshape(tf.reduce_mean(v2 - self.feg * v), [1, 1]) / self.tstep
        self.tderr = tf.reshape(tderr,(1,1))

        if plasticity:
            dwc = self.vscale * self.tstep * self.lr * tderr * tf.transpose(pre)
            self.model.layers[-2].set_weights([self.model.layers[-2].get_weights()[0] + dwc])

            ea = tf.linalg.matmul(pre, tf.cast(post, dtype=tf.float32), transpose_a=True)
            dwa = self.tstep * self.lr * tderr * ea
            self.model.layers[-1].set_weights([self.model.layers[-1].get_weights()[0] + dwa])

            self.dwa = np.sum(abs(dwa))
            self.dwc = np.sum(abs(dwc))
        return tderr, v2

    def cri_reset(self):
        self.vstate = tf.zeros([1, self.ncri])


class LSMmodel(tf.keras.Model):
    def __init__(self, hp):
        super(LSMmodel, self).__init__()
        self.nact = hp['nact']
        self.ncri = hp['ncri']
        self.crins = np.sqrt(hp['ctau']/hp['tstep']) * hp['crins']
        total_inputs = hp['npc']**2 + hp['cuesize'] + hp['fbsz']  # place cell, cue

        rnncell = RNNCell(hp=hp, ninput=total_inputs)
        self.rnn = tf.keras.layers.RNN(cell=rnncell, return_sequences=False, return_state=True, stateful=False,
                                       time_major=True)
        self.critic = tf.keras.layers.Dense(units=self.ncri, activation='linear',
                                           use_bias=False, kernel_initializer='zeros', name='critic')
        self.actor = tf.keras.layers.Dense(units=self.nact, activation='linear',
                                            use_bias=False, kernel_initializer='zeros', name='actor')

    def call(self, inputs, states):
        r, h = self.rnn(inputs, initial_state=tf.cast(states,dtype=tf.float32))
        c = self.critic(r) + tf.random.normal(shape=(1,self.ncri),stddev=self.crins)
        q = self.actor(r)
        return r, h, q, c


'''
Custom Activation
'''


def bump_activation(x):
    g01 = (x * tf.cast(tf.keras.backend.greater(x, 0), tf.float32) * tf.cast(tf.keras.backend.less_equal(x, 0.5),
                                                                             tf.float32)) ** 2
    has_nans = tf.cast(tf.sqrt(2 * x - 0.5), tf.float32) * tf.cast(tf.keras.backend.greater(x, 0.5), tf.float32)
    g1 = tf.where(tf.math.is_nan(has_nans), tf.zeros_like(has_nans), has_nans)
    return g01 + g1


def phi_b(x, threshold=0):
    g1 = x * tf.cast(tf.keras.backend.greater(x, threshold), tf.float32)
    g0 = threshold * tf.cast(tf.keras.backend.less_equal(x, threshold), tf.float32)
    return g0 + g1


def no_activation(x):
    return x


def heviside(x,thres=0):
    return 1 * tf.cast(tf.keras.backend.greater(x, thres), tf.float32)


def set_excitatory(matsize=(67,1024),K=int(67/2)):
    mat = np.random.uniform(low=-1,high=0,size=matsize)
    for i in range(matsize[1]):
        idx = np.random.choice(np.arange(matsize[0]), size=K, replace=False)
        mat[idx, i] = abs(mat[idx, i])
    return norm_ms(mat)

def norm_ms(w):
    u = np.mean(w)
    sig = np.std(w)
    nw = (w-u)/sig
    return nw

def norm_mimx(w):
    mx = np.max(w)
    mi = np.min(w)
    nw = 2*((w-mi)/(mx-mi))-1
    return nw


def choose_activation(actname,hp=None):
    if actname == 'sigm':
        act = tf.sigmoid
    elif actname == 'tanh':
        act = tf.tanh
    elif actname == 'relu':
        act = tf.nn.relu
    elif actname == 'softplus':
        act = tf.nn.softplus
    elif actname == 'elu':
        act = tf.nn.elu
    elif actname == 'leakyrelu':
        act = tf.nn.leaky_relu
    elif actname == 'ReExp':
        def ReExp(x, A=2, B=2, threshold=0, max_value=None):
            return tf.keras.activations.relu(A * tf.exp(B * x) - A, alpha=0, max_value=max_value, threshold=threshold)
        act = ReExp
    elif actname == 'pois':
        print('p0: {}, Theta: {}, du: {}, '.format(hp['p0'], hp['theta'],hp['du']))
        def InhomPois(u, p0=hp['p0'], theta=hp['theta'], du=hp['du']):
            return p0 * tf.exp((u - theta) / du)
        act = InhomPois
    elif actname == 'bump':
        act = bump_activation
    elif actname == 'phib':
        def phi_b(x, threshold=hp['sparsity']):
            g1 = x * tf.cast(tf.keras.backend.greater(x, threshold), tf.float32)
            g0 = threshold * tf.cast(tf.keras.backend.less_equal(x, threshold), tf.float32)
            return g0 + g1
        act = phi_b
    elif actname == 'hevi':
        def heviside(x, threshold=hp['sparsity']):
            return 1 * tf.cast(tf.keras.backend.greater(x, threshold), tf.float32)
        act = heviside
    elif actname == 'relu1':
        act = tf.keras.layers.ReLU(max_value=1,negative_slope=0,threshold=0)
    elif actname == 'phia':
        act = tf.keras.layers.ReLU(max_value=None, negative_slope=0, threshold=hp['sparsity'])
    elif actname == 'softmax':
        act = tf.nn.softmax
    else:
        act = no_activation
    return act


'''
Place & Action Cells
'''


class place_cells():
    def __init__(self, hp):
        self.sigcoeff = 2  # larger coeff makes distribution sharper
        self.npc = hp['npc']  # vpcn * hpcn  # square maze
        self.au = hp['mazesize']
        hori = np.linspace(-self.au / 2, self.au / 2, self.npc)
        vert = np.linspace(-self.au / 2, self.au / 2, self.npc)
        self.pcdev = hori[1] - hori[0]  # distance between each place cell

        self.pcs = np.zeros([self.npc * self.npc, 2])
        i = 0
        for x in hori[::-1]:
            for y in vert:
                self.pcs[i] = np.array([y, x])
                i += 1

    def sense(self, s):
        ''' to convert coordinate s to place cell activity '''
        norm = np.sum((s - self.pcs) ** 2, axis=1)
        pcact = np.exp(-norm / (self.sigcoeff * self.pcdev ** 2))
        return pcact

    def check_pc(self, showpc='n'):
        ''' to show place cell distribution on Maze '''
        if showpc == 'y':
            plt.figure()
            plt.scatter(self.pcs[:, 0], self.pcs[:, 1], s=20, c='r')
            plt.axis((-self.au / 2, self.au / 2, -self.au / 2, self.au / 2))
            for i in range(self.npc):
                circ = plt.Circle(self.pcs[i], self.pcdev, color='g', fill=False)
                plt.gcf().gca().add_artist(circ)
            plt.show()


class action_cells():
    def __init__(self, hp):
        self.nact = hp['nact']
        self.alat = hp['alat']
        self.tstep = hp['tstep']
        self.astep = hp['maxspeed'] * self.tstep
        thetaj = (2 * np.pi * np.arange(1, self.nact + 1)) / self.nact
        self.aj = tf.cast(self.astep * np.array([np.sin(thetaj), np.cos(thetaj)]), dtype=tf.float32)
        self.qalpha = self.tstep / hp['qtau']
        self.qstate = tf.zeros((1, self.nact))  # initialise actor units to 0
        self.ns = np.sqrt(1 / self.qalpha) * hp['actns']
        self.maxactor = deque(maxlen=500)

        wminus = hp['actorw-']  # -1
        wplus = hp['actorw+']  # 1
        psi = hp['actorpsi']  # 20
        thetaj = (2 * np.pi * np.arange(1, self.nact + 1)) / self.nact
        thetadiff = np.tile(thetaj[None, :], (self.nact, 1)) - np.tile(thetaj[:, None], (1, self.nact))
        f = np.exp(psi * np.cos(thetadiff))
        f = f - f * np.eye(self.nact)
        norm = np.sum(f, axis=0)[0]
        self.wlat = tf.cast((wminus/self.nact) + wplus * f / norm,dtype=tf.float32)
        self.actact = choose_activation(hp['actact'],hp)

    def reset(self):
        self.qstate = tf.zeros((1, self.nact)) # reset actor units to 0

    def move(self, q):
        Y = q + tf.random.normal(mean=0, stddev=self.ns, shape=(1, self.nact), dtype=tf.float32)
        if self.alat:
            Y += tf.matmul(self.actact(self.qstate),self.wlat)
        self.qstate = (1 - self.qalpha) * self.qstate + self.qalpha * Y
        rho = self.actact(self.qstate)
        at = tf.matmul(self.aj, rho, transpose_b=True).numpy()[:, 0]/self.nact

        movedist = np.linalg.norm(at,2)*1000/self.tstep  # m/s
        self.maxactor.append(movedist)
        return at, rho


'''
Bump Agents
'''
class BumpCell(tf.keras.layers.Layer):
    def __init__(self,hp):
        super(BumpCell, self).__init__()
        self.nbump = hp['nwm']
        self.balpha = hp['tstep'] / hp['btau']
        self.bumpns = np.sqrt(1 / self.balpha) * hp['bumpns']
        self.brecact = choose_activation(hp['brecact'],hp)
        self.bact = choose_activation(hp['bact'],hp)
        cueinput = hp['cuesize']

        ''' win weight init'''
        # loading weights for each cue to 3 bump units
        w_in = np.zeros((cueinput, self.nbump))
        sz = self.nbump // cueinput
        for i in range(cueinput):
            w_in[i, i*sz:(i*sz)+sz] = 1/(sz*1)

        ''' wrec weight init '''
        thetaj = (2 * np.pi * np.arange(1, self.nbump + 1)) / self.nbump
        thetadiff = np.tile(thetaj[None, :], (self.nbump, 1)) - np.tile(thetaj[:, None], (1, self.nbump))
        f = np.exp(hp['bumppsi'] * np.cos(thetadiff))
        norm = np.sum(f, axis=0)[0]
        wbump = (hp['bumpw-'] / self.nbump) + hp['bumpw+'] * f / norm
        self.wbump = np.concatenate((w_in, wbump), axis=0)

    @property
    def state_size(self):
        return self.nbump

    @property
    def output_size(self):
        return self.nrnn

    def build(self, inputs_shape):
        if inputs_shape[1] is None:
            raise ValueError("Expected inputs.shape[-1] but shape: %s" % inputs_shape)

        self._kernel = self.add_weight(
            'W_in_W_rec',
            shape=[inputs_shape[1] + self.nbump, self.nbump],
            initializer=tf.constant_initializer(self.wbump))
        self.built = True

    def call(self, inputs, state):
        bumpinput = tf.matmul(tf.concat([inputs, self.brecact(state[0])], 1), self._kernel)
        xbt = (1 - self.balpha) * state[0] + self.balpha * (bumpinput + tf.random.normal(shape=(1, self.nbump), mean=0,
                                                                                         stddev=self.bumpns))
        mt = self.bact(xbt)

        return mt, xbt


class BumpSimpleAgent:
    def __init__(self, hp, env):
        ''' Copy environment parameters '''
        self.env = env
        self.tstep = hp['tstep']

        ''' agent parameters '''
        self.nwm = hp['nwm']
        self.taug = hp['taug']
        self.beg = (1 - (self.tstep / self.taug))  # taur for backward euler continuous TD
        self.feg = (1 + (self.tstep / self.taug))  # taur for forward euler continuous TD
        self.lr = hp['lr']
        self.npc = hp['npc']
        self.nact = hp['nact']
        self.workmem = hp['workmem']
        self.actns = hp['actns']
        if hp['controltype'] == 'expand':
            self.rstate = tf.zeros([1, (67+self.nwm) * ((hp['nhid'] // (self.npc ** 2 + hp['cuesize'] + self.nwm)) + 1)])
        else:
            self.rstate = tf.zeros([1, hp['nhid']])

        ''' critic parameters '''
        self.ncri = hp['ncri']
        self.vstate = tf.zeros([1, self.ncri])
        self.vscale = hp['vscale']
        self.calpha = hp['tstep']/hp['ctau']
        self.criact = choose_activation(hp['criact'],hp)
        self.eulerm = hp['eulerm']

        ''' Setup model: Place cell --> Action cells '''
        self.pc = place_cells(hp)
        self.model = BumpSimpleModel(hp)
        self.ac = action_cells(hp)

    def act(self, state, cue_r_fb, mstate):
        s = self.pc.sense(state)  # convert coordinate info to place cell activity
        state_cue_fb = np.concatenate([s, cue_r_fb])  # combine all inputs

        if self.workmem and self.env.i <= self.env.workmemt:
            # silence state presentation during cue presentation
            state_cue_fb[:self.npc ** 2] = 0

        ''' Predict next action '''
        x, q, c, m, mh = self.model(tf.cast(state_cue_fb[None, :], dtype=tf.float32), mstate)

        return x, q, c, m, mh

    def learn(self, s1, cue_r1_fb, m1, pre, post, R, v,plasticity=True):
        ''' Hebbian rule: lr * TD * eligibility trace '''

        _, _, c2, _, _ = self.act(s1, cue_r1_fb, m1)
        self.vstate = (1 - self.calpha) * self.vstate + self.calpha * c2
        v2 = self.criact(self.vstate)

        if self.eulerm == 0:
            tderr = R + tf.reshape(tf.reduce_mean(self.beg * v2 - v),[1,1]) / self.tstep
        elif self.eulerm == 1:
            tderr = R + tf.reshape(tf.reduce_mean(v2 - self.feg * v), [1, 1]) / self.tstep
        self.tderr = tf.reshape(tderr,(1,1))

        if plasticity:
            dwc = self.vscale * self.tstep * self.lr * tderr * tf.transpose(pre)
            self.model.layers[-2].set_weights([self.model.layers[-2].get_weights()[0] + dwc])

            ea = tf.linalg.matmul(pre, tf.cast(post, dtype=tf.float32), transpose_a=True)
            dwa = self.tstep * self.lr * tderr * ea
            self.model.layers[-1].set_weights([self.model.layers[-1].get_weights()[0] + dwa])

            self.dwa = np.sum(abs(dwa))
            self.dwc = np.sum(abs(dwc))
        return tderr, v2

    def cri_reset(self):
        self.vstate = tf.zeros([1, self.ncri])


class BumpSimpleModel(tf.keras.Model):
    def __init__(self, hp):
        super(BumpSimpleModel, self).__init__()
        self.nact = hp['nact']
        self.ncri = hp['ncri']
        self.nhid = hp['nhid']
        self.npc = hp['npc']
        self.hidscale = hp['hidscale']
        self.crins = np.sqrt(hp['ctau']/hp['tstep']) * hp['crins']
        self.hidact = hp['hidact']
        self.cuesz = hp['cuesize']
        self.nwm = hp['nwm']

        wmcell = BumpCell(hp)
        self.wm = tf.keras.layers.RNN(cell=wmcell, return_sequences=False, return_state=True, stateful=False,
                                      time_major=True)

        if hp['controltype'] == 'expand':
            self.controltype = (self.nhid // (self.npc ** 2 + self.cuesz + self.nwm)) + 1   # tile factor 10 (1024) or 80 (8192)

        elif hp['controltype'] == 'hidden':
            self.controltype = tf.keras.layers.Dense(units=self.nhid , activation=choose_activation(
                self.hidact,hp), use_bias=False, name='hidden',kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1,
                                                                                                            maxval=1,
                                                                                                            seed=None))
        else:
            self.controltype = choose_activation(self.hidact)

        self.critic = tf.keras.layers.Dense(units=self.ncri, activation='linear',
                                            use_bias=False, kernel_initializer='zeros', name='critic')
        self.actor = tf.keras.layers.Dense(units=self.nact, activation='linear',
                                           use_bias=False, kernel_initializer='zeros', name='actor')

    def call(self, inputs, states):
        if isinstance(self.controltype, int):
            r = self.hidscale * tf.tile(inputs, [1, self.controltype])
        else:
            r = self.hidscale * self.controltype(inputs)
        c = self.critic(r) + tf.random.normal(shape=(1,self.ncri),stddev=self.crins)
        q = self.actor(r)
        # pass only cue input to bump attractor
        m, mh = self.wm(inputs[:, -self.cuesz:][None, :], initial_state=tf.cast(states,dtype=tf.float32))

        return r, q, c, m, mh


class BumpLSMAgent:
    def __init__(self, hp, env):
        ''' Copy environment parameters '''
        self.env = env
        self.tstep = hp['tstep']

        ''' agent parameters '''
        self.nwm = hp['nwm']
        self.taug = hp['taug']
        self.beg = (1 - (self.tstep / self.taug))  # taur for backward euler continuous TD
        self.feg = (1 + (self.tstep / self.taug))  # taur for forward euler continuous TD
        self.lr = hp['lr']
        self.npc = hp['npc']
        self.nact = hp['nact']
        self.workmem = hp['workmem']
        self.nrnn = hp['nhid']
        self.qalpha = hp['tstep']/hp['ctau']
        self.actns = hp['actns']
        self.rstate = tf.zeros([1, hp['nhid']])

        ''' critic parameters '''
        self.ncri = hp['ncri']
        self.vstate = tf.zeros([1, self.ncri])
        self.vscale = hp['vscale']
        self.calpha = hp['tstep']/hp['ctau']
        self.criact = choose_activation(hp['criact'],hp)
        self.eulerm = hp['eulerm']
        self.fbsz = hp['fbsz']

        ''' Setup model: Place cell --> Liquid Computing Model --> Action cells '''
        self.pc = place_cells(hp)
        self.model = BumpLSMmodel(hp)
        self.ac = action_cells(hp)

    def act(self, state, cue_r_fb, rstate):
        s = self.pc.sense(state)  # convert coordinate info to place cell activity
        state_cue_fb = np.concatenate([s, cue_r_fb])  # combine all inputs

        if self.workmem and self.env.i <= self.env.workmemt:
            # silence state presentation during cue presentation
            state_cue_fb[:self.npc ** 2] = 0

        ''' Model prediction '''
        rfr, h, q, c, m, mh = self.model(tf.cast(state_cue_fb[None, None, :], dtype=tf.float32), rstate)

        return q, rfr, h, c, m, mh

    def learn(self, s1, cue_r1_fb, h1, pre, post, R, v,plasticity=True):
        ''' Hebbian rule: lr * TD * eligibility trace '''

        _, _, _, c2, _, _ = self.act(s1, cue_r1_fb, h1)
        self.vstate = (1 - self.calpha) * self.vstate + self.calpha * c2
        v2 = self.criact(self.vstate)

        if self.eulerm == 0:
            tderr = R + tf.reshape(tf.reduce_mean(self.beg * v2 - v),[1,1]) / self.tstep
        elif self.eulerm == 1:
            tderr = R + tf.reshape(tf.reduce_mean(v2 - self.feg * v), [1, 1]) / self.tstep
        self.tderr = tf.reshape(tderr,(1,1))

        if plasticity:
            dwc = self.vscale * self.tstep * self.lr * tderr * tf.transpose(pre)
            self.model.layers[-2].set_weights([self.model.layers[-2].get_weights()[0] + dwc])

            ea = tf.linalg.matmul(pre, tf.cast(post, dtype=tf.float32), transpose_a=True)
            dwa = self.tstep * self.lr * tderr * ea
            self.model.layers[-1].set_weights([self.model.layers[-1].get_weights()[0] + dwa])

            self.dwa = np.sum(abs(dwa))
            self.dwc = np.sum(abs(dwc))
        return tderr, v2

    def cri_reset(self):
        self.vstate = tf.zeros([1, self.ncri])


class BumpLSMmodel(tf.keras.Model):
    def __init__(self, hp):
        super(BumpLSMmodel, self).__init__()
        self.nact = hp['nact']
        self.ncri = hp['ncri']
        self.cuesz = hp['cuesize']
        self.crins = np.sqrt(hp['ctau']/hp['tstep']) * hp['crins']

        total_inputs = hp['npc']**2 + hp['cuesize'] + hp['nwm'] + hp['fbsz']  # place cell, cue

        rnncell = RNNCell(hp, ninput=total_inputs)
        self.rnn = tf.keras.layers.RNN(cell=rnncell, return_sequences=False, return_state=True, stateful=False,
                                       time_major=True)
        wmcell = BumpCell(hp)
        self.wm = tf.keras.layers.RNN(cell=wmcell, return_sequences=False, return_state=True, stateful=False,
                                      time_major=True)
        self.critic = tf.keras.layers.Dense(units=self.ncri, activation='linear',
                                           use_bias=False, kernel_initializer='zeros', name='critic')
        self.actor = tf.keras.layers.Dense(units=self.nact, activation='linear',
                                            use_bias=False, kernel_initializer='zeros', name='actor')

    def call(self, inputs, states):
        r, h = self.rnn(inputs, initial_state=tf.cast(states[0],dtype=tf.float32))
        c = self.critic(r) + tf.random.normal(shape=(1,self.ncri),stddev=self.crins)
        q = self.actor(r)
        m, mh = self.wm(inputs[:, :, -self.cuesz:], initial_state=tf.cast(states[1],dtype=tf.float32))

        return r, h, q, c, m, mh
