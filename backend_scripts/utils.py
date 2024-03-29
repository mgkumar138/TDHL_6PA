import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from scipy.stats import ttest_1samp
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import matplotlib
import os
import multiprocessing as mp


def savefigformats(imname, fig=None):
    if fig is None:
        fig = [plt.gcf()]
    for f in fig:
        f.savefig('{}.png'.format(imname))
        f.savefig('{}.pdf'.format(imname))
        f.savefig('{}.svg'.format(imname))
        f.savefig('{}.eps'.format(imname))


def saveload(opt, name, variblelist):
    name = name + '.pickle'
    if opt == 'save':
        with open(name, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(variblelist, f)
            print('Data Saved')
            f.close()

    if opt == 'load':
        with open(name, 'rb') as f:  # Python 3: open(..., 'rb')
            var = pickle.load(f)
            print('Data Loaded')
            f.close()
        return var


def loaddata(name):
    with open(name, 'rb') as f:  # Python 3: open(..., 'rb')
        var = pickle.load(f)
        print('Data Loaded: {}'.format(name))
        f.close()
        return var


def savedata(name, variblelist):
    name = name + '.pickle'
    with open(name, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(variblelist, f)
        print('Data Saved')
        f.close()


def plot_wchng(diffw,pltidx):
    difm = np.mean(diffw,axis=0)
    difs = np.std(diffw, axis=0)
    plt.subplot(pltidx)
    mask = np.array([-3,-2,-1])
    index = ['I&R','Action','Critic']
    df = pd.DataFrame({'OPA': difm[mask,0], 'NPA': difm[mask,1], 'NM': difm[mask,2]}, index=index)
    df2 = pd.DataFrame({'OPA': difs[mask,0], 'NPA': difs[mask,1], 'NM': difs[mask,2]}, index=index)
    ax = df.plot.bar(rot=0,ax=plt.gca(), yerr=df2/diffw.shape[0])
    for i,p in enumerate(ax.patches):
        nort, norp = ttest_1samp(diffw[:,mask,:][:,i%3,i%3], 0, axis=0)
        if norp < 0.001:
            ax.text(p.get_x()+0.25,  p.get_height()+0.5, '***', size=15)
        elif norp < 0.01:
            ax.text(p.get_x()+0.25,  p.get_height()+0.5, '**', size=15)
        elif norp < 0.05:
            ax.text(p.get_x()+0.25,  p.get_height()+ 0.5, '*', size=15)


def plot_dgr(dgr,scl, pltidx, patype):
    plt.subplot(pltidx)
    dgidx = [2 * scl - 1, 9 * scl - 1, 16 * scl - 1]
    mdg = np.mean(dgr, axis=0)
    sdg = np.std(dgr, axis=0)
    index = []
    for i in dgidx:
        index.append('S {}'.format(i+1))
    df = pd.DataFrame({'Dgr':mdg},index=index)
    df2 = pd.DataFrame({'Dgr':sdg},index=index)
    ax = df.plot.bar(rot=0, ax=plt.gca(), yerr=df2 / dgr.shape[0], color='k')
    plt.axhline(y=mdg[0], color='g', linestyle='--')
    if patype == 1:
        chnc = 100/49
    else:
        chnc = 100/6
    plt.axhline(y=chnc, color='r', linestyle='--')
    plt.title('Time Spent at Correct Location (%)')
    tv,pv = ttest_1samp(dgr, chnc, axis=0)
    for i,p in enumerate(ax.patches):
        if pv[i] < 0.001:
            ax.text(p.get_x(),  p.get_height()*1.05, '***', size=15)
        elif pv[i] < 0.01:
            ax.text(p.get_x(),  p.get_height()*1.05, '**', size=15)
        elif pv[i] < 0.05:
            ax.text(p.get_x(),  p.get_height()*1.05, '*', size=15)


def plot_maps(alldyn,mvpath, hp, pltidx, npa=6):
    qdyn = alldyn[1]
    cdyn = alldyn[2]
    nonrlen = 600
    bins = 15
    qfr = np.zeros([npa, nonrlen, 40])
    cfr = np.zeros([npa, nonrlen, 1])
    coord = np.zeros([nonrlen * npa, 2])
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

    sess = [v for v in cdyn.keys() if v.startswith(list(qdyn.keys())[-1][:9])]
    for s in sess:
        c = int(s[-1])
        qfr[c - 1] = np.array(qdyn[s])[-nonrlen:]
        cfr[c - 1] = np.array(cdyn[s])[-nonrlen:]

    qfr = np.reshape(qfr, newshape=(npa * nonrlen, 40))
    cfr = np.reshape(cfr, newshape=(npa * nonrlen, 1))

    for i, s in enumerate(sess):
        st = i * nonrlen
        ed = st + nonrlen
        coord[st:ed] = mvpath[-1, i][-nonrlen:]

    from backend_scripts.model import action_cells
    from scipy.stats import binned_statistic_2d
    actor = action_cells(hp)
    qpolicy = np.matmul(actor.aj, qfr.T)

    policy[0] = binned_statistic_2d(coord[:, 0], coord[:, 1], qpolicy[0], bins=bins, statistic='sum')[0]
    policy[1] = binned_statistic_2d(coord[:, 0], coord[:, 1], qpolicy[1], bins=bins, statistic='sum')[0]
    ccells = binned_statistic_2d(coord[:, 0], coord[:, 1], cfr[:, 0], bins=bins, statistic='mean')[0]
    policy = np.nan_to_num(policy)
    ccells = np.nan_to_num(ccells)

    plt.subplot(pltidx)
    im = plt.imshow(ccells.T,aspect='auto',origin='lower')
    plt.title('Session {} value & policy map'.format(list(qdyn.keys())[-1][7:9]))
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.quiver(newx[:, 1], newx[:, 0], policy[1].reshape(bins ** 2), policy[0].reshape(bins ** 2),
                     units='xy',color='w')
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    plt.xticks([], [])
    plt.yticks([], [])


def find_cue(c):
    c = c.reshape(18,-1)[:,0]
    if np.sum(c) > 0:
        cue = np.argmax(c)+1
    else: cue = 0
    return cue


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


def save_rdyn(rdyn, mtype,t,startpos,cue, rfr):
    rfr = tf.cast(rfr,dtype=tf.float32)
    if '{}_s{}_t{}_st{}_c{}'.format(mtype, (t // 6) + 1, t, startpos[0], find_cue(cue)) in rdyn:
        rdyn['{}_s{}_t{}_st{}_c{}'.format(mtype, (t // 6) + 1, t, startpos[0], find_cue(cue))].append(rfr.numpy()[0])
    else:
        rdyn['{}_s{}_t{}_st{}_c{}'.format(mtype, (t // 6) + 1, t, startpos[0], find_cue(cue))] = []
        rdyn['{}_s{}_t{}_st{}_c{}'.format(mtype, (t // 6) + 1, t, startpos[0], find_cue(cue))].append(rfr.numpy()[0])


def get_default_hp(task, platform='laptop'):
    if task =='1pa':
        nhid = 1024
        time = 300
        trsess = 10
        evsess = 10
        workmem = False
        Rval = 1
    elif task == '6pa':
        nhid = 8192
        time = 600
        trsess = 100
        evsess = int(trsess*.1)
        workmem = False
        Rval = 1
    elif task == 'wkm':
        nhid = 8192
        time = 600
        trsess = 20
        evsess = int(trsess*.1)
        workmem = True
        Rval = 4

    hp = {
        # Environment parameters
        'task':task,
        'mazesize': 1.6, # meters
        'tstep': 100,
        'time': time,
        'workmem': workmem,
        'render': False,
        'trsess': trsess,
        'evsess': evsess,
        'platform': platform,
        'taua': 250,
        'taub': 120,
        'npa': 6,
        'Rval': Rval,

        # input parameters
        'npc': 7,
        'sensegain': 3,
        'cuesize': 18,

        # hidden parameters
        'nhid': nhid,
        'hidact': 'relu',
        'hidscale': 1,
        'controltype': 'hidden',
        'sparsity': 0,
        'K': None,

        # actor parameters:
        'nact': 40,
        'actact': 'relu',
        'alat': True,
        'actns': 0.25,
        'qtau': 150,
        'maxspeed': 0.03,
        'actorw-': -1,
        'actorw+': 1,
        'actorpsi': 20,

        # critic parameters
        'crins': 0.0001,
        'ctau': 150,
        'vscale': 1,
        'ncri': 1,
        'criact': 'relu',

        # Bump attractor parameters
        'usebump': False,
        'nwm': 54,
        'bact': 'relu',
        'brecact': 'bump',
        'btau': 150,
        'bumppsi': 300,
        'bumpw-': -0.75,
        'bumpw+': 1,
        'bumpns': 0.1,

        # reservoir parameters
        'resact': 'relu',
        'resrecact': 'tanh',
        'rtau': 150,
        'chaos': 1.5,
        'cp': [1, 1],
        'resns': 0.025,
        'fbsz': 40+1,
        'recwinscl': 1,

        # learning parameters
        'lr': 0.00001,
        'taug': 2000,
        'eulerm': 1, # euler approximation for TD error 1 - forward, 0 - backward

        # others
        'savevar': False,
        'savefig': True,
        'saveweight': False,
        'savegenvar': False,
        'modeltype': None

    }

    if hp['platform'] == 'laptop':
        matplotlib.use('tKAgg')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        hp['cpucount'] = 1
    elif hp['platform'] == 'server':
        matplotlib.use('tKAgg')
        hp['cpucount'] = 3 #mp.cpu_count()
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif hp['platform'] == 'gpu':
        #print(tf.config.list_physical_devices('GPU'))
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        ngpu = len([x.name for x in local_device_protos if x.device_type == 'GPU'])
        #matplotlib.use('Agg')
        hp['cpucount'] = ngpu
    return hp


