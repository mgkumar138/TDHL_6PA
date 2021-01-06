import numpy as np
import matplotlib.pyplot as plt


class Maze:
    def __init__(self, hp):

        ''' Define Env Parameters '''
        self.hp=hp
        self.tstep = hp['tstep']
        self.maxstep = hp['time']*(1000 // self.tstep) # max training time
        self.workmem = hp['workmem']
        self.workmemt = 5 * (1000 // self.tstep) # cue presentation time
        self.normax = 60 * (1000 // self.tstep)  # Non-rewarded probe test max time 60s
        if self.workmem:
            self.normax += self.workmemt
        self.au = 1.6
        self.rrad = 0.03
        self.testrad = 0.1
        self.stay = False
        self.rendercall = hp['render']
        self.bounpen = 0.01
        self.punish = 0  # no punishment

        ''' Define Reward location '''
        ncues = 18
        sclf = 3  # gain for cue
        self.smell = np.eye(ncues) * sclf
        self.cue_size = self.smell.shape[1]
        self.holoc = np.zeros([49,2])

        ''' create dig sites '''
        holes = np.linspace((-self.au / 2) + 0.2, (self.au / 2) - 0.2, 7)  # each reward location is 20 cm apart
        i = 0
        for x in holes[::-1]:
            for y in holes:
                self.holoc[i] = np.array([y, x])
                i+=1

        self.landmark = np.array([self.holoc[22],self.holoc[26]])

        if self.rendercall:
            plt.ion()
            fig = plt.figure(figsize=(5, 5))
            self.ax = fig.add_subplot(111)
            self.ax.axis([-self.au/2,self.au/2,-self.au/2,self.au/2])

    def make(self, mtype='train', nocue=None, noreward=None):
        self.mtype = mtype
        if mtype =='train':
            self.rlocs = np.array([self.holoc[8],self.holoc[13], self.holoc[18], self.holoc[30], self.holoc[35],self.holoc[40]])
            self.cues = self.smell[:6]
            self.totr = 6
        elif mtype == 'pre':
            self.rlocs = np.array([self.holoc[13], self.holoc[18],self.holoc[30], self.holoc[13], self.holoc[18], self.holoc[30]])
            self.cues = np.concatenate([self.smell[1:4],self.smell[1:4]],axis=0)
            self.totr = 6
        elif mtype=='1pa' or mtype=='1train':
            self.rlocs = np.array([self.holoc[0],self.holoc[0], self.holoc[0], self.holoc[0], self.holoc[0],self.holoc[0]])
            self.cues = np.tile(self.smell[0],(6,1))
            self.totr = 1
        elif mtype=='1dpa':
            self.rlocs = np.array([self.holoc[8],self.holoc[8], self.holoc[8], self.holoc[8], self.holoc[8],self.holoc[8]])
            self.cues = np.tile(self.smell[2],(6,1))
            self.totr = 1
        elif mtype=='2dpa':
            self.rlocs = np.array([self.holoc[16],self.holoc[16], self.holoc[16], self.holoc[16], self.holoc[16],self.holoc[16]])
            self.cues = np.tile(self.smell[4],(6,1))
            self.totr = 1
        elif mtype=='3dpa':
            self.rlocs = np.array([self.holoc[24],self.holoc[24], self.holoc[24], self.holoc[24], self.holoc[24],self.holoc[24]])
            self.cues = np.tile(self.smell[6],(6,1))
            self.totr = 1
        elif mtype=='4dpa':
            self.rlocs = np.array([self.holoc[32],self.holoc[32], self.holoc[32], self.holoc[32], self.holoc[32],self.holoc[32]])
            self.cues = np.tile(self.smell[8],(6,1))
            self.totr = 1
        elif mtype=='5dpa':
            self.rlocs = np.array([self.holoc[40],self.holoc[40], self.holoc[40], self.holoc[40], self.holoc[40],self.holoc[40]])
            self.cues = np.tile(self.smell[10],(6,1))
            self.totr = 1
        elif mtype=='6dpa':
            self.rlocs = np.array([self.holoc[48],self.holoc[48], self.holoc[48], self.holoc[48], self.holoc[48],self.holoc[48]])
            self.cues = np.tile(self.smell[12],(6,1))
            self.totr = 1

        self.noct = []
        if nocue:
            for i in nocue:
                self.noct.append(np.arange((i-1)*6, i*6)) # 6 trials in a session
            self.noct = np.array(self.noct).flatten().tolist()

        self.nort = []
        if noreward:
            for i in noreward:
                self.nort.append(np.arange((i-1)*6, i*6))
            self.nort = np.array(self.nort).flatten().tolist()

    def reset(self, trial):
        if trial%6 == 0: # reset order of cues presented after 6 trials
            self.ridx = np.random.choice(6, 6, replace=False)
            self.sessr = 0
        self.idx = self.ridx[trial%6]
        self.rloc = self.rlocs[self.idx]
        self.cue = self.cues[self.idx]
        self.cueidx = np.argmax(self.cue)+1
        self.x, self.startpos = randpos(self.au)
        self.reward = 0
        self.done = False
        self.i = 0
        self.stay = False
        self.tracks = []
        self.tracks.append(self.x) # include start location
        self.t = trial
        self.cordig = 0
        self.totdig = 0
        self.dgr = 0
        if trial in self.noct: self.cue = np.zeros_like(self.cue)
        self.runR = run_Rstep(self.hp)
        self.mask = list(np.arange(6))
        self.mask.remove(self.idx)
        self.d2r = np.zeros(self.totr)
        return self.x, self.cue, self.reward, self.done

    def step(self, at):
        self.i+=1  # track number of steps taken
        R = 0

        if self.i>self.workmemt and self.workmem:
            # silence cue during movement in working memory task
            cue = np.zeros_like(self.cue)
        elif self.i<=self.workmemt and self.workmem:
            # present cue during first 5 seconds, do not update agent location
            at = np.zeros_like(at)
            cue = self.cue
        else:
            # present cue at all time steps when not working memory task
            cue = self.cue

        if self.stay:
            # stay at reward location if reached target
            at = np.zeros_like(at)
        xt1 = self.x + at  # update new location

        # if ((np.sum((self.landmark - 0.1) < xt1,axis=1)==2)*(np.sum(xt1 < (self.landmark + 0.1),axis=1)==2)).any() \
        #         and self.workmem:
        #     xt1 -= at
        #     R = self.punish

        if self.workmem:
            for ldmk in self.landmark:
                if np.linalg.norm(ldmk-xt1,2)<0.1:
                    xt1 -= at
                    R = self.punish

        ax = np.concatenate([(-self.au / 2 < xt1), (self.au / 2 > xt1)]) # -xy,+xy
        if np.sum(ax)<4:
            R = self.punish
            if np.argmin(ax)>1: # if hit right or top, bounce back by 0.01
                xt1 -=at
                xt1 += self.bounpen*(ax[2:]-1)
            elif np.argmin(ax)<=1: # if hit left or bottom, bounce back by 0.01
                xt1 -=at
                xt1 -= self.bounpen*(ax[:2]-1)

        if self.t in self.nort: # non-rewarded probe trial
            reward = 0
            # time spent = location within 0.1m near reward location with no overlap of other locations
            # if ((self.rloc - self.testrad) < xt1).all() and (xt1 < (self.rloc + self.testrad)).all():
            #     self.cordig += 1
            #     self.totdig += 1

            if np.linalg.norm(self.rloc - xt1, 2) < self.testrad:
                self.cordig += 1
                self.totdig += 1

            # elif ((np.sum((self.rlocs[self.mask] - self.testrad) < xt1,axis=1)==2)*
            #       (np.sum(xt1 < (self.rlocs[self.mask] + self.testrad),axis=1)==2)).any():
            #     self.totdig += 1

            for orl in self.rlocs[self.mask]:
                if np.linalg.norm(orl-xt1,2)<self.testrad:
                    self.totdig += 1

            if self.i == self.normax:
                self.done = True
                if self.mtype == 'train':
                    # visit ratio to correct target compared to other targets
                    self.dgr = 100 * self.cordig / (self.totdig + 1e-10)
                else:
                    # visit ratio at correct target over total time
                    if self.workmem:
                        self.dgr = np.round(100 * self.cordig / (self.normax - self.workmemt), 5)
                    else:
                        self.dgr = np.round(100 * self.cordig / (self.normax), 5)

        elif self.t in self.noct: # non-cued trial
            reward = 0
            if self.i == self.normax:
                self.done=True
        else:
            # if ((self.rloc - self.rrad) < xt1).all() and (xt1 < (self.rloc + self.rrad)).all() and self.stay is False:
            #     # if reach reward, r=1 at first instance
            #     cue = self.cue
            #     R = 1
            #     self.stay = True
            #     self.sessr +=1

            if np.linalg.norm(self.rloc - xt1, 2) < self.rrad and self.stay is False:
                # if reach reward, r=1 at first instance
                cue = self.cue
                R = 1
                self.stay = True
                self.sessr +=1

            reward, self.done = self.runR.step(R)
            if self.i >= self.maxstep:
                self.done = True

        self.tracks.append(xt1)
        distr = np.linalg.norm(xt1-self.rloc,2)  # eucledian distance from reward location
        self.x = xt1

        return self.x, cue, reward, self.done, distr

    def render(self):
        if len(self.tracks)>1:
            trl = np.array(self.tracks)
            self.ax.plot(trl[:,0],trl[:,1],'k')
        plt.show()
        plt.pause(0.001)


def randpos(au):
    stpos = (au/2)*np.concatenate([np.eye(2),-1*np.eye(2)],axis=0)
    idx = np.random.choice(4,1, replace=True) # east, north, west, south
    randst = stpos[idx]
    return randst.reshape(-1), idx


class run_Rstep():
    def __init__(self,hp):
        self.rat = 0
        self.rbt = 0
        self.rt = 0
        self.taua = hp['taua']
        self.taub = hp['taub']
        self.tstep = hp['tstep']
        self.totR = 0
        self.fullR = (1 - 1e-5) * 1/self.tstep
        self.count = False

    def convR(self,rat, rbt):
        rat = (1 - (self.tstep / self.taua)) * rat
        rbt = (1 - (self.tstep / self.taub)) * rbt
        rt = (rat - rbt) / (self.taua - self.taub)
        return rat, rbt, rt

    def step(self,R):
        if R>0 and self.count is False:
            self.fullR = (1-1e-5)*R/self.tstep
            self.count = True
        self.rat += R
        self.rbt += R
        self.rat, self.rbt, self.rt = self.convR(self.rat, self.rbt)
        self.totR += self.rt
        done = False
        if self.totR>=self.fullR: # end after fullR reached or max 3 seconds
            done = True
        return self.rt, done


class MultiplePAs:
    def __init__(self,hp):
        ''' Learning 16 PAs '''
        self.hp = hp
        self.workmem = hp['workmem']
        self.tstep = hp['tstep']
        self.maxstep = hp['time']*(1000 // self.tstep) # Train max time, 1hr
        self.workmemt = 5 * (1000 // self.tstep) # cue presentation time
        self.normax = 60 * (1000 // self.tstep)  # Non-rewarded probe test max time 60s
        if self.workmem:
            self.normax += self.workmemt
        self.au = 1.6
        self.rrad = 0.03
        self.bounpen = 0.01
        self.testrad = 0.1
        self.stay = False
        self.rendercall = hp['render']

        ''' Define Reward location '''
        ncues = 18
        holes = np.linspace((-self.au/2)+0.2,(self.au/2)-0.2,7) # each reward location is 20 cm apart
        sclf = 3 # gain for cue
        self.smell = np.eye(ncues) * sclf
        self.cue_size = self.smell.shape[1]
        self.holoc = np.zeros([49,2])

        ''' create dig sites '''
        i = 0
        for x in holes[::-1]:
            for y in holes:
                self.holoc[i] = np.array([y, x])
                i+=1

        if self.rendercall:
            plt.ion()
            fig = plt.figure(figsize=(5, 5))
            self.ax = fig.add_subplot(111)
            self.ax.axis([-self.au/2,self.au/2,-self.au/2,self.au/2])

    def make(self, mtype='16PA', nocue=None, noreward=None):
        self.mtype = mtype
        if mtype == '2PA':
            self.rlocs = np.array([self.holoc[16],self.holoc[32]])
            self.cues = self.smell[:2]

        elif mtype == '4PA':
            self.rlocs = np.array([self.holoc[16], self.holoc[32], self.holoc[18], self.holoc[30]])
            self.cues = self.smell[:4]

        elif mtype =='6PA':
            self.rlocs = np.array([self.holoc[16], self.holoc[32], self.holoc[18], self.holoc[30],
                                   self.holoc[8], self.holoc[40]])
            self.cues = self.smell[:6]

        elif mtype == '8PA':
            self.rlocs = np.array([self.holoc[16], self.holoc[32], self.holoc[18], self.holoc[30],
                                   self.holoc[8], self.holoc[40], self.holoc[12], self.holoc[36]])
            self.cues = self.smell[:8]

        elif mtype == '10PA':
            self.rlocs = np.array([self.holoc[16], self.holoc[32], self.holoc[18], self.holoc[30],
                                   self.holoc[8], self.holoc[40], self.holoc[12], self.holoc[36],
                                   self.holoc[0], self.holoc[48]])
            self.cues = self.smell[:10]

        elif mtype == '12PA':
            self.rlocs = np.array([self.holoc[16], self.holoc[32], self.holoc[18], self.holoc[30],
                                   self.holoc[8], self.holoc[40], self.holoc[12], self.holoc[36],
                                   self.holoc[0], self.holoc[48], self.holoc[6], self.holoc[42]])
            self.cues = self.smell[:12]

        elif mtype == '14PA':
            self.rlocs = np.array([self.holoc[16], self.holoc[32], self.holoc[18], self.holoc[30],
                                   self.holoc[8], self.holoc[40], self.holoc[12], self.holoc[36],
                                   self.holoc[0], self.holoc[48], self.holoc[6], self.holoc[42],
                                  self.holoc[21],self.holoc[27]])
            self.cues = self.smell[:14]

        elif mtype == '16PA':
            self.rlocs = np.array([self.holoc[16], self.holoc[32], self.holoc[18], self.holoc[30],
                                   self.holoc[8], self.holoc[40], self.holoc[12], self.holoc[36],
                                   self.holoc[0], self.holoc[48], self.holoc[6], self.holoc[42],
                                   self.holoc[21],self.holoc[27], self.holoc[3],self.holoc[45]])
            self.cues = self.smell[:16]

        self.noct = []
        if nocue:
            for i in nocue:
                self.noct.append(np.arange((i-1)*6, i*6)) # 6 trials in a session
            self.noct = np.array(self.noct).flatten().tolist()

        self.nort = []
        if noreward:
            for i in noreward:
                self.nort.append(np.arange((i-1)*len(self.rlocs), i*len(self.rlocs)))
            self.nort = np.array(self.nort).flatten().tolist()

    def reset(self, trial):
        if trial%len(self.rlocs) == 0: # reset order of cues presented after 6 trials
            self.ridx = np.random.choice(len(self.rlocs), len(self.rlocs), replace=False)
            self.sessr = 0
        self.idx = self.ridx[trial%len(self.rlocs)]
        self.rloc = self.rlocs[self.idx]
        self.cue = self.cues[self.idx]
        self.cueidx = np.argmax(self.cue)+1
        self.x, self.startpos = randpos(self.au)
        self.reward = 0
        self.done = False
        self.i = 0
        self.stay = False
        self.tracks = []
        self.tracks.append(self.x) # include start location
        self.t = trial
        self.cordig = 0
        self.totdig = 0
        self.dgr = 0
        if trial in self.noct: self.cue = np.zeros_like(self.cue)
        self.runR = run_Rstep(self.hp)
        self.mask = list(np.arange(len(self.rlocs)))
        self.mask.remove(self.idx)
        return self.x, self.cue, self.reward, self.done

    def step(self, at):
        self.i+=1  # track number of steps taken
        R = 0

        if self.i>self.workmemt and self.workmem:
            # silence cue during movement in working memory task
            cue = np.zeros_like(self.cue)
        elif self.i<=self.workmemt and self.workmem:
            # present cue during first 5 seconds, do not update agent location
            at = np.zeros_like(at)
            cue = self.cue
        else:
            # present cue at all time steps when not working memory task
            cue = self.cue

        if self.stay:
            # stay at reward location if reached target
            at = np.zeros_like(at)
        xt1 = self.x + at  # update new location

        ax = np.concatenate([(-self.au / 2 < xt1), (self.au / 2 > xt1)]) # -xy,+xy
        if np.sum(ax)<4:
            if np.argmin(ax)>1: # if hit right or top, bounce back by 0.01
                xt1 -=at
                xt1 += self.bounpen*(ax[2:]-1)
            elif np.argmin(ax)<=1: # if hit left or bottom, bounce back by 0.01
                xt1 -=at
                xt1 -= self.bounpen*(ax[:2]-1)

        if self.t in self.nort: # non-rewarded probe trial
            reward = 0
            # time spent = location within 0.1m near reward location with no overlap of other locations
            if np.linalg.norm(self.rloc - xt1, 2) < self.testrad:
                self.cordig += 1
                self.totdig += 1
            for orl in self.rlocs[self.mask]:
                if np.linalg.norm(orl-xt1,2)<self.testrad:
                    self.totdig += 1
            if self.i == self.normax:
                self.done = True
                self.dgr = 100 * self.cordig / (self.totdig + 1e-10)
        elif self.t in self.noct: # non-cued trial
            reward = 0
            if self.i == self.normax:
                self.done=True
        else:
            if np.linalg.norm(self.rloc - xt1, 2) < self.rrad and self.stay is False:
                # if reach reward, r=1 at first instance
                cue = self.cue
                R = 1
                self.stay = True
                self.sessr +=1

            reward, self.done = self.runR.step(R)
            if self.i >= self.maxstep:
                self.done = True

        self.tracks.append(xt1)
        distr = np.linalg.norm(xt1-self.rloc,2) # eucledian distance away from reward location
        self.x = xt1

        return self.x, cue, reward, self.done, distr

    def render(self):
        if len(self.tracks)>1:
            trl = np.array(self.tracks)
            self.ax.plot(trl[:,0],trl[:,1],'k')
        plt.show()
        plt.pause(0.001)


class TseMaze:
    def __init__(self, hp):
        ''' 6PA task with working memory '''
        self.hp = hp
        self.tstep = hp['tstep']
        self.maxstep = hp['time']*(1000 // self.tstep) # Train max time, 1hr
        self.workmem = hp['workmem']
        self.workmemt = 5 * (1000 // self.tstep) # cue presentation time
        self.normax = 60 * (1000 // self.tstep)  # Non-rewarded probe test max time 60s
        if self.workmem:
            self.normax += self.workmemt
        self.au = 1.6
        self.rrad = 0.03
        self.testrad = 0.1
        self.stay = False
        self.rendercall = hp['render']
        self.bounpen = 0.01
        self.punish = 0 #-0.0001
        self.totpellets = 3
        self.pellet = 3

        ''' Define Reward location '''
        ncues = 18
        sclf = 3 # gain for cue
        self.smell = np.eye(ncues) * sclf
        self.cue_size = self.smell.shape[1]
        self.holoc = np.zeros([49,2])

        ''' create dig sites '''
        holes = np.linspace((-self.au / 2) + 0.2, (self.au / 2) - 0.2, 7)  # each reward location is 20 cm apart
        i = 0
        for x in holes[::-1]:
            for y in holes:
                self.holoc[i] = np.array([y, x])
                i+=1

        self.landmark = np.array([self.holoc[22],self.holoc[26]])

        if self.rendercall:
            plt.ion()
            fig = plt.figure(figsize=(5, 5))
            self.ax = fig.add_subplot(111)
            self.ax.axis([-self.au/2,self.au/2,-self.au/2,self.au/2])

    def make(self, mtype='opa', nocue=None, noreward=None):
        self.mtype = mtype
        if mtype =='train':
            self.rlocs = np.array([self.holoc[8],self.holoc[13], self.holoc[18], self.holoc[30], self.holoc[35],self.holoc[40]])
            self.cues = self.smell[:6]
            self.totr = self.nr = 6
        elif mtype == 'pre':
            self.rlocs = np.array([self.holoc[13], self.holoc[18],self.holoc[30], self.holoc[13], self.holoc[18], self.holoc[30]])
            self.cues =  np.concatenate([self.smell[1:4],self.smell[1:4]],axis=0)
            self.totr = 6
            self.nr = 3

        self.noct = []
        if nocue:
            for i in nocue:
                self.noct.append(np.arange((i-1)*self.totr, i*self.totr)) # 6 trials in a session
            self.noct = np.array(self.noct).flatten().tolist()

        self.nort = []
        if noreward:
            for i in noreward:
                self.nort.append(np.arange((i-1)*self.totr, i*self.totr))
            self.nort = np.array(self.nort).flatten().tolist()

    def reset(self, trial,pellet):
        if trial%self.totr == 0 and pellet == 0: # reset order of cues presented after 6 trials
            self.ridx = np.random.choice(self.totr, self.totr, replace=False)
            self.sessr = 0
        if pellet == 0:
            self.startx, self.startpos = randpos(self.au) # start at same box for 1st, 2md, 3rd pellet
        self.x = self.startx.copy()
        self.idx = self.ridx[trial%self.totr]
        self.rloc = self.rlocs[self.idx]
        self.cue = self.cues[self.idx]
        self.cueidx = np.argmax(self.cue)+1
        self.startcoord = self.x.copy()
        self.reward = 0
        self.done = False
        self.i = 0
        self.stay = False
        self.tracks = []
        self.tracks.append(self.x) # include start location
        self.t = trial
        self.cordig = 0
        self.totdig = 0
        self.dgr = 0
        if trial in self.noct: self.cue = np.zeros_like(self.cue)
        self.runR = run_Rstep(self.hp)
        self.mask = list(np.arange(self.totr))
        self.mask.remove(self.idx)
        self.d2r = np.zeros(self.totr)
        self.wrongrlocs = []
        return self.x, self.cue, self.reward, self.done

    def step(self, at):
        self.i+=1
        R = 0

        if self.i>self.workmemt and self.workmem:
            cue = np.zeros_like(self.cue)
        elif self.i<=self.workmemt and self.workmem:
            at = np.zeros_like(at)
            cue = self.cue # present cue during first 5 seconds, do not update state location
        else:
            cue = self.cue # present cue when workmem == n

        if self.stay:
            at = np.zeros_like(at)
        xt1 = self.x + at # update new location

        if self.workmem:
            for ldmk in self.landmark:
                if np.linalg.norm(ldmk-xt1,2)<0.1:
                    xt1 -= at
                    R = self.punish

        if self.i>self.workmemt:
            ax = np.concatenate([(-self.au / 2 < xt1), (self.au / 2 > xt1)]) # -xy,+xy
            if np.sum(ax)<4:
                R = self.punish
                if np.argmin(ax)>1: # if hit right or top, bounce back by 0.01
                    xt1 -= at
                    xt1 += self.bounpen*(ax[2:]-1)
                elif np.argmin(ax)<=1: # if hit left or bottom, bounce back by 0.01
                    xt1 -= at
                    xt1 -= self.bounpen*(ax[:2]-1)

        if self.t in self.nort: # non-rewarded probe trial
            reward = 0
            # time spent = location within 0.1m near reward location with no overlap of other locations
            if np.linalg.norm(self.rloc - xt1, 2) < self.testrad:
                self.cordig += 1
                self.totdig += 1
            for orl in self.rlocs[self.mask]:
                if np.linalg.norm(orl-xt1,2)<self.testrad:
                    self.totdig += 1

            if self.i == self.normax:
                self.done = True
                if self.mtype == 'train':
                    self.dgr = 100 * self.cordig / (self.totdig + 1e-10)
                else:
                    if self.workmem:
                        self.dgr = np.round(100 * self.cordig / (self.normax - self.workmemt), 5)
                    else:
                        self.dgr = np.round(100 * self.cordig / (self.normax), 5)
        elif self.t in self.noct: # non-cued trial
            reward = 0
            if self.i == self.normax:
                self.done=True
        else:
            if np.linalg.norm(self.rloc - xt1, 2) < self.rrad and self.stay is False:
                # if reach reward, r=1 at first instance
                cue = self.cue
                R = 1
                self.stay = True
                self.sessr +=1

            reward, self.done = self.runR.step(R)
            if self.i >= self.maxstep:
                self.done = True
                self.sessr = len(np.unique(self.wrongrlocs))

        self.tracks.append(xt1)
        distr = np.linalg.norm(xt1-self.rloc,2) # eucledian distance away from reward location
        self.x = xt1

        return self.x, cue, reward, self.done, distr

    def render(self):
        if len(self.tracks)>1:
            trl = np.array(self.tracks)
            self.ax.plot(trl[-3:,0],trl[-3:,1],'k')
        plt.show()
        plt.pause(0.001)