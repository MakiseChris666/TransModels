import numpy as np
import torch
from torch.nn.init import xavier_normal_
from torch import nn
import torch.nn.functional as F
from load_data import getDataLoaders
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from torch.cuda.amp import autocast

class TransE(nn.Module):

    def __init__(self, entityNum, relationNum, embDim, enorm = 1, rnorm = None):
        super().__init__()
        self.entityNum = entityNum
        self.relationNum = relationNum
        self.embDim = embDim

        self.entityEmb = nn.Embedding(entityNum, embDim, max_norm = enorm)
        self.relationEmb = nn.Embedding(relationNum, embDim, max_norm = rnorm)

        self.initWeight()

    def initWeight(self):
        torch.nn.init.xavier_normal_(self.entityEmb.weight)
        torch.nn.init.xavier_normal_(self.relationEmb.weight)

class TransR(nn.Module):

    def __init__(self, entityNum, relationNum, entityEmbDim, relationEmbDim, enorm = 1, rnorm = None):
        super().__init__()
        self.entityNum = entityNum
        self.relationNum = relationNum
        self.entityEmbDim = entityEmbDim
        self.relationEmbDim = relationEmbDim

        self.entityEmb = nn.Embedding(entityNum, entityEmbDim, max_norm = enorm)
        self.relationEmb = nn.Embedding(relationNum, relationEmbDim, max_norm = rnorm)

        self.relationEmbM = nn.Embedding(relationNum, entityEmbDim * relationEmbDim)

        self.initWeight()

    def initWeight(self):
        torch.nn.init.xavier_uniform_(self.entityEmb.weight)
        torch.nn.init.xavier_uniform_(self.relationEmb.weight)
        torch.nn.init.xavier_uniform_(self.relationEmbM.weight)

    def entityEmbed(self, x, r):
        x = self.entityEmb(x)
        m = self.relationEmbM(r)
        r = self.relationEmb(r)
        x = x[..., None, :] @ m.reshape(m.shape[:-1] + (self.entityEmbDim, self.relationEmbDim))
        return x.squeeze(-2), r

    def tripletEmbed(self, h, r, t):
        h = self.entityEmb(h)
        t = self.entityEmb(t)
        m = self.relationEmbM(r)
        r = self.relationEmb(r)
        h = h[..., None, :] @ m.reshape(m.shape[:-1] + (self.entityEmbDim, self.relationEmbDim))
        t = t[..., None, :] @ m.reshape(m.shape[:-1] + (self.entityEmbDim, self.relationEmbDim))
        return h.squeeze(-2), r, t.squeeze(-2)

class TransD(nn.Module):

    def __init__(self, entityNum, relationNum, entityEmbDim, relationEmbDim, enorm = 1, rnorm = 1, epnorm = 1, rpnorm = 1):
        super().__init__()
        self.entityNum = entityNum
        self.relationNum = relationNum
        self.entityEmbDim = entityEmbDim
        self.relationEmbDim = relationEmbDim

        self.entityEmb = nn.Embedding(entityNum, entityEmbDim, max_norm = enorm)
        self.relationEmb = nn.Embedding(relationNum, relationEmbDim, max_norm = rnorm)
        self.entityEmbP = nn.Embedding(entityNum, entityEmbDim, max_norm = epnorm)
        self.relationEmbP = nn.Embedding(relationNum, relationEmbDim, max_norm = rpnorm)

    def entityEmbed(self, x, r):
        xp = self.entityEmbP(x)
        x = self.entityEmb(x)
        rp = self.relationEmbP(r)
        r = self.relationEmb(r)
        m = rp[..., :, None] @ xp[..., None, :] + torch.eye(self.relationEmbDim, self.entityEmbDim)[None, ...].to(x.device)
        x = m @ x[..., None]
        return x.squeeze(-1), r

    def entityEmbReduce(self, x, r):
        xp = self.entityEmbP(x)
        x = self.entityEmb(x)
        rp = self.relationEmbP(r)
        r = self.relationEmb(r)
        xx = xp[..., None, :] @ x[..., None]
        xx = rp * xx[..., 0] # (batch, rdim)
        xx = xx + x[..., :self.relationEmbDim]
        return xx, r

    def tripletEmbed(self, h, r, t):
        hp = self.entityEmbP(h)
        h = self.entityEmb(h)
        rp = self.relationEmbP(r)
        r = self.relationEmb(r)
        tp = self.entityEmbP(t)
        t = self.entityEmb(t)
        mrh = rp[..., :, None] @ hp[..., None, :] + torch.eye(self.relationEmbDim, self.entityEmbDim)[None, ...].to(h.device) # (batch, rdim, edim)
        mrt = rp[..., :, None] @ tp[..., None, :] + torch.eye(self.relationEmbDim, self.entityEmbDim)[None, ...].to(h.device)
        h = mrh @ h[..., None]
        t = mrt @ t[..., None]
        return h.squeeze(-1), r, t.squeeze(-1)

    def tripletEmbReduce(self, h, r, t):
        hp = self.entityEmbP(h)
        h = self.entityEmb(h)
        rp = self.relationEmbP(r)
        r = self.relationEmb(r)
        tp = self.entityEmbP(t)
        t = self.entityEmb(t)
        hh = hp[..., None, :] @ h[..., None]
        hh = rp * hh[..., 0]
        hh = hh + h[..., :self.relationEmbDim]
        tt = tp[..., None, :] @ t[..., None]
        tt = rp * tt[..., 0]
        tt = tt + t[..., :self.relationEmbDim]
        return hh, r, tt

    def initWeight(self):
        xavier_normal_(self.entityEmb.weight)
        xavier_normal_(self.relationEmb.weight)
        xavier_normal_(self.relationEmbP.weight)
        xavier_normal_(self.entityEmbP.weight)

def getHRTDict(data, entityNum):
    etoi = {e: i for i, e in enumerate(data.entities)}
    rtoi = {r: i for i, r in enumerate(data.relations)}
    hr = {}
    tr = {}
    for h, r, t in data.train_data:
        h, r, t = etoi[h], rtoi[r], etoi[t]
        if (h, r) not in hr:
            hr[(h, r)] = {t}
        else:
            if t not in hr[(h, r)]:
                hr[(h, r)].add(t)

        if (t, r) not in tr:
            tr[(t, r)] = {h}
        else:
            if h not in tr[(t, r)]:
                tr[(t, r)].add(h)

    return hr, tr

import random

class TripletSampler:

    def __init__(self, entityNum, sampleNum = 3):
        self.sampleNum = sampleNum
        self.entityNum = entityNum


    def __call__(self, triplets, *args, **kwargs):
        # triplets: (batch, 3)
        neg = torch.tile(triplets[:, None, :], (1, self.sampleNum, 1))
        r = (torch.rand(triplets.shape[0], self.sampleNum) > 0.5).long().to(neg.device) # noqa
        offset = torch.randint(1, self.entityNum, (triplets.shape[0], self.sampleNum)).long().to(neg.device)
        neg[:, :, 0] += r * offset
        neg[:, :, 2] += (1 - r) * offset
        neg[:, :, [0, 2]] %= self.entityNum

        # for i in range(neg.shape[0]):
        #     for j in range(self.sampleNum):
        #         r = random.random()
        #         if r < 0.5:
        #             while neg[i, j, 0] == triplets[i, 0]:
        #                 neg[i, j, 0] = random.choice(list(range(self.entityNum)))
        #         else:
        #             while neg[i, j, 2] == triplets[i, 2]:
        #                 neg[i, j, 2] = random.choice(list(range(self.entityNum)))
        return neg

class TripletSampler2:

    def __init__(self, entityNum, relationNum, data, sampleNum = 3):
        self.sampleNum = sampleNum
        self.entityNum = entityNum
        self.relationNum = relationNum
        self.hr, self.tr = getHRTDict(data, entityNum)
        self.candidate = list(range(entityNum))

    def __call__(self, triplets, *args, **kwargs):
        # triplets: (batch, 3)
        neg = torch.tile(triplets[:, None, :], (1, self.sampleNum, 1))
        # neg = torch.zeros((triplets.shape[0], self.sampleNum, triplets.shape[1]), dtype = torch.long).to(triplets.device)
        for i in range(neg.shape[0]):
            for j in range(self.sampleNum):
                # h, r, t = (random.randint(0, self.entityNum - 1),
                #            random.randint(0, self.relationNum - 1), random.randint(0, self.entityNum - 1))
                # if (h, r) in self.hr:
                #     while t in self.hr[(h, r)]:
                #         t = random.randint(0, self.entityNum - 1)
                # neg[i, j, 0] = h
                # neg[i, j, 1] = r
                # neg[i, j, 2] = t

                r = random.random()
                if r < 0.5:
                    blk = self.tr[(triplets[i, 2].item(), triplets[i, 1].item())]
                    cur = random.choice(self.candidate)
                    while cur in blk:
                        cur = random.choice(self.candidate)
                    neg[i, j, 0] = cur
                else:
                    blk = self.hr[(triplets[i, 0].item(), triplets[i, 1].item())]
                    cur = random.choice(self.candidate)
                    while cur in blk:
                        cur = random.choice(self.candidate)
                    neg[i, j, 2] = cur
        return neg

import time
class TransENet(nn.Module):

    def __init__(self, entityNum, relationNum, embDim, sampleNum = 3, margin = 1.0):
        super().__init__()
        self.margin = margin
        self.disfunc = nn.PairwiseDistance()
        self.sampler = TripletSampler(entityNum, sampleNum)
        self.trans = TransE(entityNum, relationNum, embDim)

    def entityEmb(self, x):
        return self.trans.entityEmb(x)

    def relationEmb(self, x):
        return self.trans.relationEmb(x)

    def forward(self, triplets):
        neg = self.sampler(triplets) # (batch, 3, 3)

        posvec1 = self.trans.entityEmb(triplets[:, 0])
        posvec2 = self.trans.relationEmb(triplets[:, 1])
        posvec3 = self.trans.entityEmb(triplets[:, 2])

        negvec1 = self.trans.entityEmb(neg[:, :, 0]) # (batch, 3, dim)
        negvec2 = self.trans.relationEmb(neg[:, :, 1])
        negvec3 = self.trans.entityEmb(neg[:, :, 2])

        posdis = self.disfunc(posvec1 + posvec2, posvec3)
        negdis = self.disfunc(negvec1 + negvec2, negvec3)
        negdis = torch.mean(negdis, dim = 1)

        return F.relu(posdis - negdis + self.margin).mean()

class TransRNet(nn.Module):

    def __init__(self, entityNum, relationNum, data, entityEmbDim, relationEmbDim, sampleNum = 3, margin = 1.0):
        super().__init__()
        self.margin = margin
        self.disfunc = nn.PairwiseDistance()
        self.sampler = TripletSampler2(entityNum, relationNum, data, sampleNum)
        self.trans = TransR(entityNum, relationNum, entityEmbDim, relationEmbDim)

    def entityEmb(self, x, r):
        return self.trans.entityEmbed(x, r)

    def relationEmb(self, x):
        return self.trans.relationEmb(x)

    def forward(self, triplets):
        neg = self.sampler(triplets) # (batch, 3, 3)
        triplets = triplets.cuda()
        neg = neg.cuda()

        posvec1, posvec2, posvec3 = self.trans.tripletEmbed(triplets[:, 0], triplets[:, 1], triplets[:, 2])

        negvec1, negvec2, negvec3 = self.trans.tripletEmbed(neg[:, :, 0], neg[:, :, 1], neg[:, :, 2]) # (batch, 3, dim)

        posdis = self.disfunc(posvec1 + posvec2, posvec3)
        negdis = self.disfunc(negvec1 + negvec2, negvec3)
        negdis = torch.mean(negdis, dim = 1)

        return F.relu(posdis - negdis + self.margin).mean()

class TransENet2(TransENet):

    def __init__(self, entityNum, relationNum, embDim, data, sampleNum = 3, margin = 1.0):
        super().__init__(entityNum, relationNum, embDim, sampleNum, margin)
        self.sampler = TripletSampler2(entityNum, relationNum, data, sampleNum)

    def forward(self, triplets):
        neg = self.sampler(triplets) # (batch, 3, 3)
        triplets = triplets.cuda()
        neg = neg.cuda()

        posvec1 = self.trans.entityEmb(triplets[:, 0])
        posvec2 = self.trans.relationEmb(triplets[:, 1])
        posvec3 = self.trans.entityEmb(triplets[:, 2])

        negvec1 = self.trans.entityEmb(neg[:, :, 0]) # (batch, 3, dim)
        negvec2 = self.trans.relationEmb(neg[:, :, 1])
        negvec3 = self.trans.entityEmb(neg[:, :, 2])

        posdis = self.disfunc(posvec1 + posvec2, posvec3)
        negdis = self.disfunc(negvec1 + negvec2, negvec3)
        negdis = torch.mean(negdis, dim = 1)

        return F.relu(posdis - negdis + self.margin).mean()

class TransDNet(nn.Module):

    def __init__(self, entityNum, relationNum, entityEmbDim, relationEmbDim, sampleNum = 3, margin = 1.0):
        super().__init__()
        self.margin = margin
        self.disfunc = nn.PairwiseDistance()
        self.sampler = TripletSampler(entityNum, sampleNum)
        self.trans = TransD(entityNum, relationNum, entityEmbDim, relationEmbDim)

    def forward(self, triplets):
        neg = self.sampler(triplets)

        posvecs = self.trans.tripletEmbReduce(triplets[:, 0], triplets[:, 1], triplets[:, 2])
        negvecs = self.trans.tripletEmbReduce(neg[..., 0], neg[..., 1], neg[..., 2])

        posdis = self.disfunc(posvecs[0] + posvecs[1], posvecs[2])
        negdis = self.disfunc(negvecs[0] + negvecs[1], negvecs[2])
        negdis = torch.mean(negdis, dim = 1)

        return F.relu(posdis - negdis + self.margin).mean()

class TransDNet2(nn.Module):

    def __init__(self, entityNum, relationNum, entityEmbDim, relationEmbDim, data, sampleNum = 3, margin = 1.0):
        super().__init__()
        self.margin = margin
        self.disfunc = nn.PairwiseDistance()
        self.sampler = TripletSampler2(entityNum, relationNum, data, sampleNum)
        self.trans = TransD(entityNum, relationNum, entityEmbDim, relationEmbDim)

    def forward(self, triplets):
        neg = self.sampler(triplets)
        triplets = triplets.cuda()
        neg = neg.cuda()

        posvecs = self.trans.tripletEmbReduce(triplets[:, 0], triplets[:, 1], triplets[:, 2])
        negvecs = self.trans.tripletEmbReduce(neg[..., 0], neg[..., 1], neg[..., 2])

        posdis = self.disfunc(posvecs[0] + posvecs[1], posvecs[2])
        negdis = self.disfunc(negvecs[0] + negvecs[1], negvecs[2])
        negdis = torch.mean(negdis, dim = 1)

        return F.relu(posdis - negdis + self.margin).mean()


from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
def trainTransE(data, args):

    trainLoader, valLoader, testLoader = getDataLoaders(data, args)
    model = TransENet(len(data.entities), len(data.relations), args.edim, sampleNum = args.sample, margin = args.margin)
    device = 'cuda' if args.cuda else 'cpu'
    model = model.to(device)

    epochs = args.num_iterations
    opt = SGD(model.parameters(), lr = args.lr)
    scheduler = ExponentialLR(opt, args.dr)

    now = datetime.now()
    timestr = now.strftime('%m%d_%H%M')
    writer = SummaryWriter('./log/' + 'TransE_' + timestr)

    for epoch in range(epochs):

        lossSum = 0.
        model.train()
        for it, x in tqdm(enumerate(trainLoader), desc = f'Training Epoch {epoch + 1}', total = len(trainLoader)):
            x = x.to(device)
            opt.zero_grad()
            loss = model(x)
            loss.backward()
            opt.step()

            lossSum += loss.item()

        scheduler.step()

        print('Epoch', epoch + 1, 'loss', lossSum / len(trainLoader))
        writer.add_scalar('loss', lossSum / len(trainLoader), epoch)

        entityVecs = model.trans.entityEmb.weight # (entityNum, embDim)
        hit10 = 0
        hit3 = 0
        hit1 = 0
        ranksum = 0
        rankpsum = 0
        model.eval()

        with torch.no_grad():

            for it, x in tqdm(enumerate(valLoader), desc = f'Validation Epoch {epoch + 1}', total = len(valLoader)):
                x = x.to(device)
                predVec = model.entityEmb(x[:, 0]) + model.relationEmb(x[:, 1]) # (batch, embDim)
                dists = torch.sqrt(torch.sum((entityVecs[None, :, :] - predVec[:, None, :]) ** 2, dim = -1)) # (batch, entityNum)
                sortedDists, sortedDistsI = torch.sort(dists, dim = 1)
                ranks = torch.where(sortedDistsI == x[:, [2]])[1] + 1 # noqa
                ranksum += torch.sum(ranks).item()
                rankpsum += torch.sum(1 / ranks).item()
                hit10 += torch.sum(ranks <= 10).item()
                hit3 += torch.sum(ranks <= 3).item()
                hit1 += torch.sum(ranks == 1).item()

            print('Validation Result:')
            print('Hits @10: %.4f' % (hit10 / len(data.valid_data)))
            print('Hits @3: %.4f' % (hit3 / len(data.valid_data)))
            print('Hits @1: %.4f' % (hit1 / len(data.valid_data)))
            print('MR: %.4f' % (ranksum / len(data.valid_data)))
            print('MRR: %.4f' % (rankpsum / len(data.valid_data)))

            writer.add_scalar('val hit10', hit10 / len(data.valid_data), epoch)
            writer.add_scalar('val hit3', hit3 / len(data.valid_data), epoch)
            writer.add_scalar('val hit1', hit1 / len(data.valid_data), epoch)
            writer.add_scalar('val MR', ranksum / len(data.valid_data), epoch)
            writer.add_scalar('val MRR', rankpsum / len(data.valid_data), epoch)

            if epoch % 2 == 1:
                hit10 = 0
                hit3 = 0
                hit1 = 0
                ranksum = 0
                rankpsum = 0
                for it, x in tqdm(enumerate(testLoader), desc = f'Test Epoch {epoch + 1}', total = len(testLoader)):
                    x = x.to(device)
                    predVec = model.entityEmb(x[:, 0]) + model.relationEmb(x[:, 1])  # (batch, embDim)
                    dists = torch.sqrt(
                        torch.sum((entityVecs[None, :, :] - predVec[:, None, :]) ** 2, dim = -1))  # (batch, entityNum)
                    sortedDists, sortedDistsI = torch.sort(dists, dim = 1)
                    ranks = torch.where(sortedDistsI == x[:, [2]])[1] + 1  # noqa
                    ranksum += torch.sum(ranks).item()
                    rankpsum += torch.sum(1 / ranks).item()
                    hit10 += torch.sum(ranks <= 10).item()
                    hit3 += torch.sum(ranks <= 3).item()
                    hit1 += torch.sum(ranks == 1).item()

                print('Test Result:')
                print('Hits @10: %.4f' % (hit10 / len(data.test_data)))
                print('Hits @3: %.4f' % (hit3 / len(data.test_data)))
                print('Hits @1: %.4f' % (hit1 / len(data.test_data)))
                print('MR: %.4f' % (ranksum / len(data.test_data)))
                print('MRR: %.4f' % (rankpsum / len(data.test_data)))

                writer.add_scalar('test hit10', hit10 / len(data.test_data), epoch)
                writer.add_scalar('test hit3', hit3 / len(data.test_data), epoch)
                writer.add_scalar('test hit1', hit1 / len(data.test_data), epoch)
                writer.add_scalar('test MR', ranksum / len(data.test_data), epoch)
                writer.add_scalar('test MRR', rankpsum / len(data.test_data), epoch)

def trainTransR(data, args):

    metrics = dict(
        valid = [],
        test = []
    )  # (hit10, mr, mrr)

    trainLoader, valLoader, testLoader = getDataLoaders(data, args)
    model = TransRNet(len(data.entities), len(data.relations), data, args.edim, args.rdim, sampleNum = args.sample, margin = args.margin)
    device = 'cuda' if args.cuda else 'cpu'
    model = model.to(device)

    epochs = args.num_iterations
    opt = SGD(model.parameters(), lr = args.lr)
    scheduler = ExponentialLR(opt, args.dr)

    now = datetime.now()
    timestr = now.strftime('%m%d_%H%M')
    writer = SummaryWriter('./log/' + 'TransR_' + timestr)

    for epoch in range(epochs):

        lossSum = 0.
        model.train()
        for it, x in tqdm(enumerate(trainLoader), desc = f'Training Epoch {epoch + 1}', total = len(trainLoader)):
            # x = x.to(device)
            opt.zero_grad()
            loss = model(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            opt.step()

            lossSum += loss.item()

        scheduler.step()

        print('Epoch', epoch + 1, 'loss', lossSum / len(trainLoader))
        writer.add_scalar('loss', lossSum / len(trainLoader), epoch)


        hit10 = 0
        hit3 = 0
        hit1 = 0
        ranksum = 0
        rankpsum = 0
        model.eval()

        with torch.no_grad():

            evecs = []
            evec = torch.arange(len(data.entities)).to(device)
            rvec = torch.zeros(len(data.entities), dtype = torch.long).to(device)
            for ri in range(len(data.relations)):
                rvec[...] = ri
                evecs.append(model.entityEmb(evec, rvec)[0][None, ...])
            evecs = torch.cat(evecs, dim = 0)

            for it, x in tqdm(enumerate(valLoader), desc = f'Validation Epoch {epoch + 1}', total = len(valLoader)):
                x = x.to(device)
                hvec, rvec = model.entityEmb(x[:, 0], x[:, 1])
                predVec = hvec + rvec # (batch, embDim)
                entityVecs = evecs[x[:, 1]]
                dists = torch.sqrt(torch.sum((entityVecs - predVec[:, None, :]) ** 2, dim = -1)) # (batch, entityNum)
                sortedDists, sortedDistsI = torch.sort(dists, dim = 1)
                ranks = torch.where(sortedDistsI == x[:, [2]])[1] + 1 # noqa
                ranksum += torch.sum(ranks).item()
                rankpsum += torch.sum(1 / ranks).item()
                hit10 += torch.sum(ranks <= 10).item()
                hit3 += torch.sum(ranks <= 3).item()
                hit1 += torch.sum(ranks == 1).item()

            print('Validation Result:')
            print('Hits @10: %.4f' % (hit10 / len(data.valid_data)))
            print('Hits @3: %.4f' % (hit3 / len(data.valid_data)))
            print('Hits @1: %.4f' % (hit1 / len(data.valid_data)))
            print('MR: %.4f' % (ranksum / len(data.valid_data)))
            print('MRR: %.4f' % (rankpsum / len(data.valid_data)))

            writer.add_scalar('val hit10', hit10 / len(data.valid_data), epoch)
            writer.add_scalar('val hit3', hit3 / len(data.valid_data), epoch)
            writer.add_scalar('val hit1', hit1 / len(data.valid_data), epoch)
            writer.add_scalar('val MR', ranksum / len(data.valid_data), epoch)
            writer.add_scalar('val MRR', rankpsum / len(data.valid_data), epoch)

            metrics['valid'].append(
                (hit10 / len(data.valid_data), ranksum / len(data.valid_data), rankpsum / len(data.valid_data)))

            if epoch % 2 == 1:
                hit10 = 0
                hit3 = 0
                hit1 = 0
                ranksum = 0
                rankpsum = 0
                for it, x in tqdm(enumerate(testLoader), desc = f'Test Epoch {epoch + 1}', total = len(testLoader)):
                    x = x.to(device)
                    hvec, rvec = model.entityEmb(x[:, 0], x[:, 1])
                    predVec = hvec + rvec  # (batch, embDim)
                    entityVecs = evecs[x[:, 1]]
                    dists = torch.sqrt(
                        torch.sum((entityVecs - predVec[:, None, :]) ** 2, dim = -1))  # (batch, entityNum)
                    sortedDists, sortedDistsI = torch.sort(dists, dim = 1)
                    ranks = torch.where(sortedDistsI == x[:, [2]])[1] + 1  # noqa
                    ranksum += torch.sum(ranks).item()
                    rankpsum += torch.sum(1 / ranks).item()
                    hit10 += torch.sum(ranks <= 10).item()
                    hit3 += torch.sum(ranks <= 3).item()
                    hit1 += torch.sum(ranks == 1).item()

                print('Test Result:')
                print('Hits @10: %.4f' % (hit10 / len(data.test_data)))
                print('Hits @3: %.4f' % (hit3 / len(data.test_data)))
                print('Hits @1: %.4f' % (hit1 / len(data.test_data)))
                print('MR: %.4f' % (ranksum / len(data.test_data)))
                print('MRR: %.4f' % (rankpsum / len(data.test_data)))

                writer.add_scalar('test hit10', hit10 / len(data.test_data), epoch)
                writer.add_scalar('test hit3', hit3 / len(data.test_data), epoch)
                writer.add_scalar('test hit1', hit1 / len(data.test_data), epoch)
                writer.add_scalar('test MR', ranksum / len(data.test_data), epoch)
                writer.add_scalar('test MRR', rankpsum / len(data.test_data), epoch)

                metrics['test'].append(
                    (hit10 / len(data.valid_data), ranksum / len(data.valid_data), rankpsum / len(data.valid_data)))

    return metrics

def trainTransD(data, args):

    trainLoader, valLoader, testLoader = getDataLoaders(data, args)
    model = TransDNet(len(data.entities), len(data.relations), args.edim, args.rdim, sampleNum = args.sample, margin = args.margin)
    device = 'cuda' if args.cuda else 'cpu'
    model = model.to(device)

    epochs = args.num_iterations
    opt = SGD(model.parameters(), lr = args.lr, momentum = 0.9)
    scheduler = ExponentialLR(opt, args.dr)

    now = datetime.now()
    timestr = now.strftime('%m%d_%H%M')
    writer = SummaryWriter('./log/' + 'TransD_' + timestr)

    for epoch in range(epochs):

        lossSum = 0.
        model.train()
        for it, x in tqdm(enumerate(trainLoader), desc = f'Training Epoch {epoch + 1}', total = len(trainLoader)):
            x = x.to(device)
            opt.zero_grad()
            loss = model(x)
            loss.backward()
            opt.step()

            lossSum += loss.item()

        scheduler.step()

        print('Epoch', epoch + 1, 'loss', lossSum / len(trainLoader))
        writer.add_scalar('loss', lossSum / len(trainLoader), epoch)

        hit10 = 0
        hit3 = 0
        hit1 = 0
        ranksum = 0
        rankpsum = 0
        model.eval()

        with torch.no_grad():

            evecps = model.trans.entityEmbP.weight # (enum, edim)
            rvecps = model.trans.relationEmbP.weight # (rnum, rdim)
            evec = model.trans.entityEmb.weight # (enum, edim)
            evecs = [] # (rnum, enum, rdim)
            for ri in range(len(data.relations)):
                # mres = rvecps[ri, None, :, None] @ evecps[:, None, :] + torch.eye(args.rdim, args.edim).to(device) # (enum, rdim, edim)
                # revec = mres @ evec[:, :, None] # (enum, rdim)

                revec = evecps[:, None, :] @ evec[:, :, None] # (enum, 1, 1)
                revec = rvecps[ri, None, :] * revec[..., 0] # (enum, rdim)
                evecs.append(revec[None, ...])
            evecs = torch.cat(evecs, dim = 0)

            for it, x in tqdm(enumerate(valLoader), desc = f'Validation Epoch {epoch + 1}', total = len(valLoader)):
                x = x.to(device)
                hvec, rvec = model.trans.entityEmbReduce(x[:, 0], x[:, 1])
                predVec = hvec + rvec # (batch, rdim)
                entityVecs = evecs[x[:, 1]] # (batch, enum, rdim)
                dists = torch.sqrt(
                    torch.sum((entityVecs - predVec[:, None, :]) ** 2, dim = -1))  # (batch, entityNum)
                sortedDists, sortedDistsI = torch.sort(dists, dim = 1)
                ranks = torch.where(sortedDistsI == x[:, [2]])[1] + 1  # noqa
                ranksum += torch.sum(ranks).item()
                rankpsum += torch.sum(1 / ranks).item()
                hit10 += torch.sum(ranks <= 10).item()
                hit3 += torch.sum(ranks <= 3).item()
                hit1 += torch.sum(ranks == 1).item()

            print('Validation Result:')
            print('Hits @10: %.4f' % (hit10 / len(data.valid_data)))
            print('Hits @3: %.4f' % (hit3 / len(data.valid_data)))
            print('Hits @1: %.4f' % (hit1 / len(data.valid_data)))
            print('MR: %.4f' % (ranksum / len(data.valid_data)))
            print('MRR: %.4f' % (rankpsum / len(data.valid_data)))

            writer.add_scalar('val hit10', hit10 / len(data.valid_data), epoch)
            writer.add_scalar('val hit3', hit3 / len(data.valid_data), epoch)
            writer.add_scalar('val hit1', hit1 / len(data.valid_data), epoch)
            writer.add_scalar('val MR', ranksum / len(data.valid_data), epoch)
            writer.add_scalar('val MRR', rankpsum / len(data.valid_data), epoch)

            if epoch % 2 == 1:
                hit10 = 0
                hit3 = 0
                hit1 = 0
                ranksum = 0
                rankpsum = 0
                for it, x in tqdm(enumerate(testLoader), desc = f'Test Epoch {epoch + 1}', total = len(testLoader)):
                    x = x.to(device)
                    hvec, rvec = model.trans.entityEmbed(x[:, 0], x[:, 1])
                    predVec = hvec + rvec  # (batch, rdim)
                    entityVecs = evecs[x[:, 1]]  # (batch, enum, rdim)
                    dists = torch.sqrt(
                        torch.sum((entityVecs - predVec[:, None, :]) ** 2, dim = -1))  # (batch, entityNum)
                    sortedDists, sortedDistsI = torch.sort(dists, dim = 1)
                    ranks = torch.where(sortedDistsI == x[:, [2]])[1] + 1  # noqa
                    ranksum += torch.sum(ranks).item()
                    rankpsum += torch.sum(1 / ranks).item()
                    hit10 += torch.sum(ranks <= 10).item()
                    hit3 += torch.sum(ranks <= 3).item()
                    hit1 += torch.sum(ranks == 1).item()

                print('Test Result:')
                print('Hits @10: %.4f' % (hit10 / len(data.test_data)))
                print('Hits @3: %.4f' % (hit3 / len(data.test_data)))
                print('Hits @1: %.4f' % (hit1 / len(data.test_data)))
                print('MR: %.4f' % (ranksum / len(data.test_data)))
                print('MRR: %.4f' % (rankpsum / len(data.test_data)))

                writer.add_scalar('test hit10', hit10 / len(data.test_data), epoch)
                writer.add_scalar('test hit3', hit3 / len(data.test_data), epoch)
                writer.add_scalar('test hit1', hit1 / len(data.test_data), epoch)
                writer.add_scalar('test MR', ranksum / len(data.test_data), epoch)
                writer.add_scalar('test MRR', rankpsum / len(data.test_data), epoch)

def trainTransE2(data, args):

    metrics = dict(
        valid = [],
        test = []
    ) # (hit10, mr, mrr)

    trainLoader, valLoader, testLoader = getDataLoaders(data, args)
    model = TransENet2(len(data.entities), len(data.relations), args.edim, data, sampleNum = args.sample, margin = args.margin)
    device = 'cuda' if args.cuda else 'cpu'
    model = model.to(device)

    epochs = args.num_iterations
    opt = SGD(model.parameters(), lr = args.lr)
    scheduler = ExponentialLR(opt, args.dr)

    now = datetime.now()
    timestr = now.strftime('%m%d_%H%M')
    writer = SummaryWriter('./log/' + 'TransE_' + timestr)

    for epoch in range(epochs):

        lossSum = 0.
        model.train()
        for it, x in tqdm(enumerate(trainLoader), desc = f'Training Epoch {epoch + 1}', total = len(trainLoader)):
            # x = x.to(device)
            opt.zero_grad()
            loss = model(x)
            loss.backward()
            opt.step()

            lossSum += loss.item()

        scheduler.step()

        print('Epoch', epoch + 1, 'loss', lossSum / len(trainLoader))
        writer.add_scalar('loss', lossSum / len(trainLoader), epoch)

        entityVecs = model.trans.entityEmb.weight # (entityNum, embDim)
        hit10 = 0
        hit3 = 0
        hit1 = 0
        ranksum = 0
        rankpsum = 0
        model.eval()

        with torch.no_grad():

            for it, x in tqdm(enumerate(valLoader), desc = f'Validation Epoch {epoch + 1}', total = len(valLoader)):
                x = x.to(device)
                predVec = model.entityEmb(x[:, 0]) + model.relationEmb(x[:, 1]) # (batch, embDim)
                dists = torch.sqrt(torch.sum((entityVecs[None, :, :] - predVec[:, None, :]) ** 2, dim = -1)) # (batch, entityNum)
                sortedDists, sortedDistsI = torch.sort(dists, dim = 1)
                ranks = torch.where(sortedDistsI == x[:, [2]])[1] + 1 # noqa
                ranksum += torch.sum(ranks).item()
                rankpsum += torch.sum(1 / ranks).item()
                hit10 += torch.sum(ranks <= 10).item()
                hit3 += torch.sum(ranks <= 3).item()
                hit1 += torch.sum(ranks == 1).item()

            print('Validation Result:')
            print('Hits @10: %.4f' % (hit10 / len(data.valid_data)))
            print('Hits @3: %.4f' % (hit3 / len(data.valid_data)))
            print('Hits @1: %.4f' % (hit1 / len(data.valid_data)))
            print('MR: %.4f' % (ranksum / len(data.valid_data)))
            print('MRR: %.4f' % (rankpsum / len(data.valid_data)))

            writer.add_scalar('val hit10', hit10 / len(data.valid_data), epoch)
            writer.add_scalar('val hit3', hit3 / len(data.valid_data), epoch)
            writer.add_scalar('val hit1', hit1 / len(data.valid_data), epoch)
            writer.add_scalar('val MR', ranksum / len(data.valid_data), epoch)
            writer.add_scalar('val MRR', rankpsum / len(data.valid_data), epoch)

            metrics['valid'].append((hit10 / len(data.valid_data), ranksum / len(data.valid_data), rankpsum / len(data.valid_data)))

            if epoch % 2 == 1:
                hit10 = 0
                hit3 = 0
                hit1 = 0
                ranksum = 0
                rankpsum = 0
                for it, x in tqdm(enumerate(testLoader), desc = f'Test Epoch {epoch + 1}', total = len(testLoader)):
                    x = x.to(device)
                    predVec = model.entityEmb(x[:, 0]) + model.relationEmb(x[:, 1])  # (batch, embDim)
                    dists = torch.sqrt(
                        torch.sum((entityVecs[None, :, :] - predVec[:, None, :]) ** 2, dim = -1))  # (batch, entityNum)
                    sortedDists, sortedDistsI = torch.sort(dists, dim = 1)
                    ranks = torch.where(sortedDistsI == x[:, [2]])[1] + 1  # noqa
                    ranksum += torch.sum(ranks).item()
                    rankpsum += torch.sum(1 / ranks).item()
                    hit10 += torch.sum(ranks <= 10).item()
                    hit3 += torch.sum(ranks <= 3).item()
                    hit1 += torch.sum(ranks == 1).item()

                print('Test Result:')
                print('Hits @10: %.4f' % (hit10 / len(data.test_data)))
                print('Hits @3: %.4f' % (hit3 / len(data.test_data)))
                print('Hits @1: %.4f' % (hit1 / len(data.test_data)))
                print('MR: %.4f' % (ranksum / len(data.test_data)))
                print('MRR: %.4f' % (rankpsum / len(data.test_data)))

                writer.add_scalar('test hit10', hit10 / len(data.test_data), epoch)
                writer.add_scalar('test hit3', hit3 / len(data.test_data), epoch)
                writer.add_scalar('test hit1', hit1 / len(data.test_data), epoch)
                writer.add_scalar('test MR', ranksum / len(data.test_data), epoch)
                writer.add_scalar('test MRR', rankpsum / len(data.test_data), epoch)

                metrics['test'].append(
                    (hit10 / len(data.valid_data), ranksum / len(data.valid_data), rankpsum / len(data.valid_data)))

    return metrics

def trainTransD2(data, args):

    metrics = dict(
        valid = [],
        test = []
    )  # (hit10, mr, mrr)

    trainLoader, valLoader, testLoader = getDataLoaders(data, args)
    model = TransDNet2(len(data.entities), len(data.relations), args.edim, args.rdim, data, sampleNum = args.sample, margin = args.margin)
    device = 'cuda' if args.cuda else 'cpu'
    model = model.to(device)

    epochs = args.num_iterations
    opt = SGD(model.parameters(), lr = args.lr)
    scheduler = ExponentialLR(opt, args.dr)

    now = datetime.now()
    timestr = now.strftime('%m%d_%H%M')
    writer = SummaryWriter('./log/' + 'TransD_' + timestr)

    for epoch in range(epochs):

        lossSum = 0.
        model.train()
        for it, x in tqdm(enumerate(trainLoader), desc = f'Training Epoch {epoch + 1}', total = len(trainLoader)):
            # x = x.to(device)
            opt.zero_grad()
            loss = model(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            opt.step()

            lossSum += loss.item()

        scheduler.step()

        print('Epoch', epoch + 1, 'loss', lossSum / len(trainLoader))
        writer.add_scalar('loss', lossSum / len(trainLoader), epoch)

        hit10 = 0
        hit3 = 0
        hit1 = 0
        ranksum = 0
        rankpsum = 0
        model.eval()

        with torch.no_grad():

            evecps = model.trans.entityEmbP.weight # (enum, edim)
            rvecps = model.trans.relationEmbP.weight # (rnum, rdim)
            evec = model.trans.entityEmb.weight # (enum, edim)
            evecs = [] # (rnum, enum, rdim)
            for ri in range(len(data.relations)):
                # mres = rvecps[ri, None, :, None] @ evecps[:, None, :] + torch.eye(args.rdim, args.edim).to(device) # (enum, rdim, edim)
                # revec = mres @ evec[:, :, None] # (enum, rdim)

                revec = evecps[:, None, :] @ evec[:, :, None] # (enum, 1, 1)
                revec = rvecps[ri, None, :] * revec[..., 0] # (enum, rdim)
                evecs.append(revec[None, ...])
            evecs = torch.cat(evecs, dim = 0)

            for it, x in tqdm(enumerate(valLoader), desc = f'Validation Epoch {epoch + 1}', total = len(valLoader)):
                x = x.to(device)
                hvec, rvec = model.trans.entityEmbReduce(x[:, 0], x[:, 1])
                predVec = hvec + rvec # (batch, rdim)
                entityVecs = evecs[x[:, 1]] # (batch, enum, rdim)
                dists = torch.sqrt(
                    torch.sum((entityVecs - predVec[:, None, :]) ** 2, dim = -1))  # (batch, entityNum)
                sortedDists, sortedDistsI = torch.sort(dists, dim = 1)
                ranks = torch.where(sortedDistsI == x[:, [2]])[1] + 1  # noqa
                ranksum += torch.sum(ranks).item()
                rankpsum += torch.sum(1 / ranks).item()
                hit10 += torch.sum(ranks <= 10).item()
                hit3 += torch.sum(ranks <= 3).item()
                hit1 += torch.sum(ranks == 1).item()

            print('Validation Result:')
            print('Hits @10: %.4f' % (hit10 / len(data.valid_data)))
            print('Hits @3: %.4f' % (hit3 / len(data.valid_data)))
            print('Hits @1: %.4f' % (hit1 / len(data.valid_data)))
            print('MR: %.4f' % (ranksum / len(data.valid_data)))
            print('MRR: %.4f' % (rankpsum / len(data.valid_data)))

            writer.add_scalar('val hit10', hit10 / len(data.valid_data), epoch)
            writer.add_scalar('val hit3', hit3 / len(data.valid_data), epoch)
            writer.add_scalar('val hit1', hit1 / len(data.valid_data), epoch)
            writer.add_scalar('val MR', ranksum / len(data.valid_data), epoch)
            writer.add_scalar('val MRR', rankpsum / len(data.valid_data), epoch)

            metrics['valid'].append(
                (hit10 / len(data.valid_data), ranksum / len(data.valid_data), rankpsum / len(data.valid_data)))

            if epoch % 2 == 1:
                hit10 = 0
                hit3 = 0
                hit1 = 0
                ranksum = 0
                rankpsum = 0
                for it, x in tqdm(enumerate(testLoader), desc = f'Test Epoch {epoch + 1}', total = len(testLoader)):
                    x = x.to(device)
                    hvec, rvec = model.trans.entityEmbed(x[:, 0], x[:, 1])
                    predVec = hvec + rvec  # (batch, rdim)
                    entityVecs = evecs[x[:, 1]]  # (batch, enum, rdim)
                    dists = torch.sqrt(
                        torch.sum((entityVecs - predVec[:, None, :]) ** 2, dim = -1))  # (batch, entityNum)
                    sortedDists, sortedDistsI = torch.sort(dists, dim = 1)
                    ranks = torch.where(sortedDistsI == x[:, [2]])[1] + 1  # noqa
                    ranksum += torch.sum(ranks).item()
                    rankpsum += torch.sum(1 / ranks).item()
                    hit10 += torch.sum(ranks <= 10).item()
                    hit3 += torch.sum(ranks <= 3).item()
                    hit1 += torch.sum(ranks == 1).item()

                print('Test Result:')
                print('Hits @10: %.4f' % (hit10 / len(data.test_data)))
                print('Hits @3: %.4f' % (hit3 / len(data.test_data)))
                print('Hits @1: %.4f' % (hit1 / len(data.test_data)))
                print('MR: %.4f' % (ranksum / len(data.test_data)))
                print('MRR: %.4f' % (rankpsum / len(data.test_data)))

                writer.add_scalar('test hit10', hit10 / len(data.test_data), epoch)
                writer.add_scalar('test hit3', hit3 / len(data.test_data), epoch)
                writer.add_scalar('test hit1', hit1 / len(data.test_data), epoch)
                writer.add_scalar('test MR', ranksum / len(data.test_data), epoch)
                writer.add_scalar('test MRR', rankpsum / len(data.test_data), epoch)

                metrics['test'].append(
                    (hit10 / len(data.valid_data), ranksum / len(data.valid_data), rankpsum / len(data.valid_data)))

    return metrics


if __name__ == '__main__':
    model = TransD(8, 8, 8, 8)
    h = torch.arange(8)
    r = torch.arange(8)
    t = torch.arange(7, -1, -1)

    h1, r1, t1 = model.tripletEmbed(h, r, t)
    h2, r2, t2 = model.tripletEmbReduce(h, r, t)
    print(h1.shape, r1.shape, t1.shape)
    print(h2.shape, r2.shape, t2.shape)

    print(h1 - h2, r1 - r2, t1 - t2)