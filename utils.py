from scipy.stats import zscore
import numpy as np
import scipy.stats as ss
from scipy.interpolate import interp1d

def smooth_line(inputx, inputy):
    xx = np.linspace(inputx.min(),inputx.max(), 1000)
    itp = interp1d(inputx,inputy, kind='linear')
    window_size, poly_order = 301, 2
    yy = savgol_filter(itp(xx), window_size, poly_order) # smooth
    return xx, yy

def remove_false(inputx, inputy):
    inputy = np.array(inputy)
    inputx = np.array(inputx)
    valid_ind = np.where(inputy > 0)[0]
    return inputx[valid_ind], inputy[valid_ind]

def important_axons(stim_w, proportion=0.1):
    # thresh = 0.001
    # axons_contribution = np.mean(stim_weights[:, 45:], axis=1)
    # axons_contribution_prelick = np.mean(prelick_weights[:, :45], axis=1)
    # idx = np.where(axons_contribution > thresh)[0]
    # non_idx = np.where(axons_contribution <= thresh)[0]
    NA = stim_weights.shape[0]
    axons_contribution = np.mean(stim_weights[:, 45:], axis=1)
    axons_contribution_threshold = np.mean(axons_contribution) + np.std(axons_contribution)
    w_sort = np.argsort(axons_contribution)
    NNA = int(proportion*NA)
    idx = w_sort[-NNA:]
    non_idx = w_sort[:(NA-NNA)]
#     idx = np.where(axons_contribution > axons_contribution_threshold)
#     non_idx = np.where(axons_contribution <= axons_contribution_threshold)
    return idx, non_idx

def classifier(x, y, ratio=2/3):
    '''
    ratio: train:test
    '''
    accs = []
    weights = []
    # make sure each class has same number of samples
    ind1 = np.where(y==0)[0]
    ind2 = np.where(y==1)[0]
    valid_amount = min(len(ind2), len(ind1))
    if valid_amount < 10:
        print('no avalible binary labels for classifier')
        return False, False
    valid_ind1 = ind1[np.random.choice(np.arange(len(ind1)), valid_amount, replace=False)]
    valid_ind2 = ind2[np.random.choice(np.arange(len(ind2)), valid_amount, replace=False)]
    valid_ind = np.hstack([valid_ind1, valid_ind2])
    print('valid samples: {}'.format(len(valid_ind)))
    x = x[valid_ind]
    y = y[valid_ind]
    N = len(x) # number of samples
    F = x.shape[1] # number of features
    for seed in range(10):
        random.seed(seed)
        np.random.seed(seed)
        itrain = np.random.choice(np.arange(N), size=int(N*ratio), replace=False)
        itest = np.setxor1d(np.arange(N), itrain)
        ylabel = y[itrain]
        new_seed = 1000
        while (np.sum(ylabel[0] == ylabel) == len(ylabel)):
            new_seed +=1
            random.seed(seed+new_seed)
            np.random.seed(seed+new_seed)
            itrain = np.random.choice(np.arange(N), size=int(N*ratio), replace=False)
            itest = np.setxor1d(np.arange(N), itrain)
            ylabel = y[itrain]
            if new_seed >= 1050:
                print('no avalible binary labels for classifier')
                return False, False

        model = LogisticRegression(solver='liblinear', random_state=seed)
        model.fit(x[itrain], y[itrain])
        te_pred = model.predict(x[itest])
        acc = np.sum(te_pred == y[itest]) / len(itest)
        cur_weights = model.coef_
        accs.append(acc)
        weights.append(cur_weights)
    avg_acc = np.mean(accs)
    weights = np.mean(np.array(weights), axis=0)
    return avg_acc, weights

def behavior_performance(mouse_name, drive):
    beh_data_path = "{}:/Jenni/{}/behavior/justbehavior".format(drive, mouse_name)
    fnames = [_ for _ in os.listdir(beh_data_path) if _.endswith('txt')]

    from tkinter import Tcl
    fnames = Tcl().call('lsort', '-dict', fnames)
    nBlocks = len(fnames)
    pbRes = np.zeros((4, nBlocks))
    rlRes = np.zeros((4, nBlocks))
    nPB = np.zeros(nBlocks)
    days = np.zeros(nBlocks)
    for i in range(nBlocks):
        tmp = fnames[i].split('_')[-1]
        days[i] = int(tmp.split('v')[0])
        fpath = os.path.join(beh_data_path, fnames[i])
        bmat = load_behavior_txt(fpath)
        behavior = bmat[:, 12].reshape(-1).astype(int)
        outcome = bmat[:, 3].reshape(-1).astype(int)
        nPB[i] = np.sum(behavior==0)
        if nPB[i] > 0:
            pbIdx = np.where(behavior==0)
            pbOutcome = outcome[pbIdx]
            for j in range(4):
                pbRes[j, i] = np.sum(pbOutcome == j+1) / nPB[i]

        rlIdx = np.where(behavior==1)
        rlOutcome = outcome[rlIdx]
        for j in range(4):
            rlRes[j, i] = np.sum(rlOutcome == j+1) / (100 - nPB[i])
    days = days.astype('int')
    x = np.arange(nBlocks)
    ptrue = pbRes[0] + pbRes[3]
    rtrue = rlRes[0] + rlRes[3]
    return ptrue, rtrue, fnames

def plot_weight(day_groups, valid_days, weights, accs, title='', smooth=False, figname=False):
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 8
    cmap = plt.cm.get_cmap('viridis_r')

    plt.figure(figsize=(6, 4))
    for i, days in enumerate(day_groups):
        wsum = 0
        ndays = 0
        for d in days:
            idx = np.where(np.array(valid_days) == d)[0][0]
            # weight = hf_weights[idx]
            weight = np.abs(weights[idx])
            if accs[idx]:
                ndays +=1
                # wsum += np.mean(weight, axis=0)
                wsum += (np.mean(weight, axis=0) - np.mean(weight))
        wavg = wsum/ndays
        if smooth:
            xx, yy = smooth_line(np.arange(len(wavg)), wavg)
            plt.plot(xx, yy, label='period {}'.format(i), color=cmap(i/len(day_groups)))
        else:
            plt.plot(np.arange(len(wavg)), wavg, label='period {}'.format(i), color=cmap(i/len(day_groups)))
        # plt.axvline(45, color='gray', linestyle=":", alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.xlabel('frame')
    plt.ylabel('avearge weight')
    if figname:
        plt.savefig(figname)
        # plt.close()
        
def plot_avg_weight(day_groups, valid_days, weights, accs, title='', smooth=False, figname=False):
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 8
    cmap = plt.cm.get_cmap('viridis_r')

    plt.figure(figsize=(6, 4))
    for i, days in enumerate(day_groups):
        wsum = 0
        ndays = 0
        for d in days:
            idx = np.where(np.array(valid_days) == d)[0][0]
            if accs[idx]:
                ndays +=1
                wsum += weights[idx].squeeze()
        wavg = wsum/ndays
        if smooth:
            xx, yy = smooth_line(np.arange(len(wavg)), wavg)
            plt.plot(xx, yy, label='period {}'.format(i), color=cmap(i/len(day_groups)))
        else:
            plt.plot(np.arange(len(wavg)), wavg, label='period {}'.format(i), color=cmap(i/len(day_groups)))
        # plt.axvline(45, color='gray', linestyle=":", alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.xlabel('frame')
    plt.ylabel('avearge weight')
    if figname:
        plt.savefig(figname)
        # plt.close()

def load_behavior_txt(fpath):
    res = np.zeros((100, 13))
    with open(fpath,'r') as f:
        for nline, line in enumerate(f.readlines()):
            line = line.strip()
            row = line.split(',')
            for k, v in enumerate(row):
                res[nline, k] = v
    return res

def RCM(features):
    '''
    Compute Representational Correlation Matrix given the features
    :param features: an array of features, should be (N,D), N features with dimension D.
    :return: (N,N) distance matrix
    '''
    z = ss.zscore(features, axis=1)
    D = features.shape[-1]
    corrs = np.matmul(z, z.T) / D
    return corrs

def daily_blocks(data, day=1):
    'return block indexs of a specific day, only return valid block indexes'
    iblocks = []
    nBlocks = data.shape[-1]
    for i in range(nBlocks):
        block = data[0, i]
        d = block['day']
        if len(d[0]) > 0:
            d = d[0][0]
            if d == day:
                iblocks.append(i)
    return iblocks

def block_stim_evoke_response(block, window=[30, 45], normalize=False, dff=True, remove=False):
    '''
    normalize: zscore or not
    dff: If True, use df/f to calculate stim evoked resposne. If False, use original response to get stim evoded response
    remove: if true, remove continouse non-licking trials(set number of continous trials to be 20)
    '''
    raw_spks = block['imagingdata']
    NA, NT = raw_spks.shape
    if normalize:
        spks = zscore(raw_spks, axis=1)
        # spks = sigmoid(spks)
        nan_flag = np.isnan(np.sum(spks, axis=1))
        not_nan_ind = np.where(nan_flag==False)[0]
        spks = spks[not_nan_ind]
    else:
        spks = raw_spks

    behavior = block['behavdata']
    istim = np.where(behavior[0] > 0)[0]
    cues = behavior[0, istim]
    outcomes = behavior[1, istim]
    ifirstlicks = -np.ones(len(istim))
    licks = behavior[3]

    all_df = []
    all_licks = []
    if dff:
        for i,k in enumerate(istim):
            ilick = np.where(behavior[2, (k - window[0]):(k + window[1])] > 0)[0]
            if len(ilick)>0:
                ifirstlicks[i] = int(np.where(behavior[2, (k - window[0]):(k + window[1])] > 0)[0])  # index of first lick
            f0 = np.mean(spks[:, (k - 10):k], axis=1)
            ft = spks[:, (k - window[0]):(k + window[1])]
            f0 = np.repeat(f0[:, np.newaxis], ft.shape[-1], axis=1)
            df = (ft - f0) / f0
            all_df.append(df)
            all_licks.append(licks[(k - window[0]):(k + window[0])])
    else:
        for i,k in enumerate(istim):
            ilick = np.where(behavior[2, (k - window[0]):(k + window[1])] > 0)[0]
            if len(ilick)>0:
                ifirstlicks[i] = int(np.where(behavior[2, (k - window[0]):(k + window[1])] > 0)[0])  # index of first lick
            ft = spks[:, (k - window[0]):(k + window[1])]
            all_df.append(ft)
            all_licks.append(licks[(k - window[0]):(k + window[1])])
    all_df = np.stack(all_df)
    all_licks = np.stack(all_licks)
    if remove:
        nonlick_flag = (outcomes == 4) | (outcomes == 2) # CR and Miss
        remove_flag = np.ones(nonlick_flag.shape)
        for i in range(20, len(nonlick_flag)):
            if np.sum(nonlick_flag[(i-20):i]) == 20:
                remove_flag[(i-20):i] = 0
        remove_flag = (remove_flag == 1)
        outcomes = outcomes[remove_flag]
        cues = cues[remove_flag]
        ifirstlicks = ifirstlicks[remove_flag]
        all_df = all_df[remove_flag]
        all_licks = all_licks[remove_flag]
        print('remove {}/{} trials with window size 20'.format(len(nonlick_flag) - np.sum(remove_flag), len(nonlick_flag)))
    return cues, outcomes, ifirstlicks, all_df, all_licks


def get_all_tone_response(block, context='rl'):
    '''
    return average spk of each trial
    context: 'rl' or 'pb', 'rl' is default
    '''
    ttspks = []  # target tone response
    ftspks = []  # foil tone response

    raw_spks = block['imagingdata']
    NA, NT = raw_spks.shape
    spks = raw_spks

    if NT > 0:
        behavior = block['behavdata']
        icondition = np.where(behavior[0] < 3)[0]  # Rl condition indexes

        if context == 'rl':
            # target tone
            itt = np.where(behavior[0] == 1)[0]
            # foil tone
            ift = np.where(behavior[0] == 2)[0]
        elif context == 'pb':
            itt = np.where(behavior[0] == 3)[0]
            ift = np.where(behavior[0] == 4)[0]
        else:
            printn("only support 'rl' or 'pb' as context!")

        ## 2 outcomes
        for i in range(2):
            n = 0
            oidx = np.where(behavior[1] == (i + 1))[0]  # outcome idx
            oidx = np.intersect1d(itt, oidx)  # overlap of tone  context and outcome idx
            all_df = []
            for k in oidx:
                if (100 <= k) and ((k + 100) <= NT):
                    f0 = np.mean(spks[:, (k - 30):(k - 15)], axis=1)
                    ft = spks[:, (k - 30):(k + 45)]
                    f0 = np.repeat(f0[:, np.newaxis], ft.shape[-1], axis=1)
                    df = (ft - f0) / f0
                    all_df.append(np.mean(df, axis=0))
                    n += 1
            if n > 0:
                all_df = np.vstack(all_df)
            ttspks.append(all_df)

            n = 0
            oidx = np.where(behavior[1] == (4 - i))[0]  # outcome idx
            oidx = np.intersect1d(ift, oidx)  # overlap of ton and outcome idx
            all_df = []
            for k in oidx:
                if (100 <= k) and ((k + 100) <= NT):
                    f0 = np.mean(spks[:, (k - 30):(k - 15)], axis=1)
                    ft = spks[:, (k - 30):(k + 45)]
                    f0 = np.repeat(f0[:, np.newaxis], ft.shape[-1], axis=1)
                    df = (ft - f0) / f0
                    all_df.append(np.mean(df, axis=0))
                    n += 1
            if n > 0:
                all_df = np.vstack(all_df)
            ftspks.append(all_df)
        spks_list = [ttspks[0], ttspks[1], ftspks[0], ftspks[1]]  # [hit_spk, miss_spk, cr_spk, fa_apk]
        return spks_list
    else:
        return None

    
def extract_all_licks(block, disc):
    behavior = block['behavdata']
    # find the alllick txt file and behavior file
    fname = block['behaviorfilename'][0]
    print('processing {} ...'.format(fname))
    mouse_name = fname.split('_')[0]
    
    dlick = np.loadtxt("{}/{}/behavior/{}.txtlicks.txt".format(disc, mouse_name, fname))
    dframes = np.loadtxt('{}/{}/behavior/{}.txtframes.txt'.format(disc, mouse_name, fname))
    diff = dlick[1:] - dlick[:-1]
    startlickind = np.where(diff>0)[0] + 1 # index in dframes
    testframes = np.zeros(len(dframes));
    diff = dframes[1:] - dframes[:-1]
    framestmp = np.where(diff>0)[0] + 1 # where the non-zero frame start

    iframe = 1
    for i in range(len(framestmp)-1):
        framelen = framestmp[i+1]-framestmp[i]
        if framelen < 20:
            testframes[framestmp[i]:framestmp[i+1]] = iframe*np.ones(framestmp[i+1]-framestmp[i])
            iframe += 1
        else:
            testframes[:framestmp[i+1]] = -1
            iframe = 1
    print('number of frames in total:', testframes.max())
    print('number of frames in total(2 plane):', int(testframes.max()/2))
    print('number of frames in structure: {}'.format(behavior.shape[-1]))
    allframediff = np.sign(int(testframes.max()/2) - behavior.shape[-1]) # result - real
    if allframediff == 0:
        allframediff = 1
    # load all licks
    ilickstart = testframes[startlickind]
    ilickstart = ilickstart[np.where(ilickstart > 0)]
    print('total number of licks: ', len(ilickstart))

    alllickframes = np.floor(ilickstart/2) - 1 # 2 planes
    realfirstlick = np.where(behavior[2] > 0)[0]
    print('number of tones: ', len(realfirstlick))

    
    # correct the mismatches and only preserve the firstlick frame for the mismatch > 1
    print('mismatch frames:')
    raw_alllickframes = alllickframes.copy()
    _, imatchrealfirstlick, _ = np.intersect1d(realfirstlick, alllickframes, return_indices=True)
    imismatches = np.setdiff1d(np.arange(len(realfirstlick)), imatchrealfirstlick)
    mismatchframes = realfirstlick[imismatches]

    matchfirstframes, imatchrealfirstlick, ialllickframes = np.intersect1d(realfirstlick, alllickframes, return_indices=True)
    imismatches = np.setdiff1d(np.arange(len(realfirstlick)), imatchrealfirstlick)
    mismatchframes = realfirstlick[imismatches]
    finalmismatches = []
    for i, mismatchframe in enumerate(mismatchframes):
        inext = np.where(realfirstlick==mismatchframe)[0][0] + 1
        if inext < (len(realfirstlick)-1):
            nextframe = realfirstlick[inext]
        else:
            nextframe = -1
            
        if (mismatchframe+allframediff) in raw_alllickframes:
            idx = np.where(raw_alllickframes == (mismatchframe+allframediff))[0]
            istart = idx[0]
            if nextframe == -1:
                alllickframes[istart:] -= allframediff
            else: 
                idx = np.where(raw_alllickframes < nextframe)[0]
                iend = idx[-1]
                alllickframes[istart:iend] -= allframediff
        elif (mismatchframe-allframediff) in raw_alllickframes:
            idx = np.where(raw_alllickframes == (mismatchframe-allframediff))[0]
            istart = idx[0]
            if nextframe == -1:
                alllickframes[istart:] += allframediff
            else: 
                idx = np.where(raw_alllickframes < nextframe)[0]
                iend = idx[-1]
                alllickframes[istart:iend] += allframediff
        else:
            idx = np.where(raw_alllickframes > mismatchframe)[0]
            print(mismatchframe, raw_alllickframes[(idx[0]-1):(idx[0]+1)])
    
    alllickframes = alllickframes.astype('int')
    alllicks = np.zeros(behavior.shape[-1])
    for i, ilick in enumerate(realfirstlick):
        if i == (len(realfirstlick)-1):
            iend = -1
        else:
            nextlick = realfirstlick[i+1]
            iend = np.where(alllickframes < nextlick)[0]
            iend = iend[-1]
        if ilick in alllickframes:
            istart = np.where(alllickframes==ilick)[0][0]
            if iend == -1:
                alllicks[alllickframes[istart:]] = 1
            else:
                alllicks[alllickframes[istart:iend]] = 1
            # alllicks[alllickframes[istart:(istart + 20)]] = 1
        else:
            alllicks[ilick] = 1
    # return behavior matrix with the 4th row to be the all licks.
    behavior[3] = alllicks
    print('{} finished!\n'.format(fname))
    return behavior, realfirstlick, raw_alllickframes


def get_tone_response_df(block, context='rl'):
    '''
    context: 'rl' or 'pb', 'rl' is default
    '''
    ttspks = []  # target tone response
    ftspks = []  # foil tone response

    raw_spks = block['imagingdata']
    NA, NT = raw_spks.shape
    spks = raw_spks

    if NT > 0:
        behavior = block['behavdata']
        icondition = np.where(behavior[0] < 3)[0]  # Rl condition indexes
        #         spks = raw_spks - raw_spks.mean(axis=1)[:, np.newaxis]
        #         spks = spks / (spks**2).mean(axis=1)[:, np.newaxis]**.5

        if context == 'rl':
            # target tone
            itt = np.where(behavior[0] == 1)[0]
            # foil tone
            ift = np.where(behavior[0] == 2)[0]
        elif context == 'pb':
            itt = np.where(behavior[0] == 3)[0]
            ift = np.where(behavior[0] == 4)[0]
        else:
            printn("only support 'rl' or 'pb' as context!")

        ## 2 outcomes
        for i in range(2):
            avg_df = 0
            n = 0
            oidx = np.where(behavior[1] == (i + 1))[0]  # outcome idx
            oidx = np.intersect1d(itt, oidx)  # overlap of ton and outcome idx
            for k in oidx:
                if (100 <= k) and ((k + 100) <= NT):
                    f0 = np.mean(spks[:, (k - 30):(k - 15)], axis=1)
                    ft = spks[:, (k - 30):(k + 45)]
                    f0 = np.repeat(f0[:, np.newaxis], ft.shape[-1], axis=1)
                    df = (ft - f0) / f0
                    avg_df += df
                    n += 1
            if n > 0:
                avg_df = avg_df / n
            else:
                avg_df = []
            ttspks.append(avg_df)

            avg_df = 0
            n = 0
            oidx = np.where(behavior[1] == (4 - i))[0]  # outcome idx
            oidx = np.intersect1d(ift, oidx)  # overlap of ton and outcome idx
            for k in oidx:
                if (100 <= k) and ((k + 100) <= NT):
                    f0 = np.mean(spks[:, (k - 30):(k - 15)], axis=1)
                    ft = spks[:, (k - 30):(k + 45)]
                    f0 = np.repeat(f0[:, np.newaxis], ft.shape[-1], axis=1)
                    df = (ft - f0) / f0
                    avg_df += df
                    n += 1
            if n > 0:
                avg_df = avg_df / n
            else:
                avg_df = []
            ftspks.append(avg_df)

    spks_list = [ttspks[0], ttspks[1], ftspks[0], ftspks[1]]
    # [hit_spk, miss_spk, cr_spk, fa_apk]
    return spks_list