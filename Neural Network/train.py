# This script was developed on Bazzite Linux 7.2.15-ogc1.1.fc45.x86_64
# Gemini 3.1 Pro assisted in debugging and code cleanup (see comments)
# Handouts from CSMC 191 Neural Networks, as well as the book 'Neural Networks From Scratch In Python" by Kinsley & Kukiela
# 	were used to assist in the creation of this script
#
# Author: Borromeo, Daniel William R.
# Date Created  : 17 November 2026
# Date Modified : 23 November 2026
# Version       : 1.2.0
import glob, os, random, math, argparse, csv
from neuralnet_v1 import NeuralNetwork

FEATURES = ['ghi', 'ghi_iav', 'ghi_season', 'ghi_low', 't', 'trange', 'rh', 'ps', 'ws', 'ws99', 'rain', 'wet']

def read_power(path):
    with open(path) as fh: lines = fh.read().splitlines()
    start = next(i for i, l in enumerate(lines)  if l.startswith('-END HEADER-')) + 1
    cols = lines[start].split(',')
    rows = [dict(zip(cols, map(float, l.split(',')))) for l in lines[start+1:] if l.strip()]

    return [r for r in rows if r['YEAR'] < 2026 and all(v != -999 for v in r.values())]

def mean(v):
    return sum(v)/len(v)

def std(v):
    m = mean(v)
    return math.sqrt(sum((x-m)**2 for x in v)/ len(v))

def quantile(v, q):
    s = sorted(v)
    return s[min(int(q * len(s)), len(s) - 1)]

def climate_profile(path):
    d = read_power(path)
    n = len(d)
    g = [r['ALLSKY_SFC_SW_DWN'] for r in d]

    yearly, monthly, yRain = {}, {}, {}
    for r in d:
        yearly.setdefault(r['YEAR'], []).append(r['ALLSKY_SFC_SW_DWN'])
        monthly.setdefault(r['MO'], []).append(r['ALLSKY_SFC_SW_DWN'])
        yRain[r['YEAR']] = yRain.get(r['YEAR'], 0) + r['PRECTOTCORR']
    ym = [mean(v) for v in yearly.values()]
    mm = [mean(v) for v in monthly.values()]

    return [
        mean(g), std(ym),
        (max(mm) - min(mm)) / mean(mm),
        sum(x < 2 for x in g) / n,
        mean([r['T2M'] for r in d]),
        mean([r['T2M_RANGE'] for r in d]),
        mean([r['RH2M'] for r in d]), 
        mean([r['PS'] for r in d]),
        mean([r['WS10M'] for r in d]),
        quantile([r['WS10M'] for r in d], 0.99),
        mean(list(yRain.values())),
        sum(r['PRECTOTCORR'] > 1 for r in d) / n
    ]

def load_folder(folder):
    return {
        os.path.basename(p)[:-4]: climate_profile(p) for p in sorted(glob.glob(os.path.join(folder, '*.csv')))
    }

def make_scaler(x):
    lo = [min(c) for c in zip(*x)]
    hi = [max(c) for c in zip(*x)]

    return lo, hi

def scale(x, lo, hi):
    return[min(1.0, max(0.0, 0.1+0.8*(v - l) / (h - l))) if h > l else 0.5 for v, l, h in zip(x, lo, hi)]

def auc(scores, labels):
    P = [s for s, l in zip(scores, labels) if l == 1]
    N = [s for s, l in zip(scores, labels) if l == 0]

    return sum((p > n) + 0.5 * (p==n) for p in P for n in N) / (len(P) * len(N))

def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(base_dir, '..', 'Datasets')
    outputs_dir = os.path.join(base_dir, '..', 'Outputs')

    ap = argparse.ArgumentParser()
    ap.add_argument('--pos', default=os.path.join(datasets_dir, 'Weather (SF)'))
    ap.add_argument('--neg', default=os.path.join(datasets_dir, 'Weather (NSF)'))
    ap.add_argument('--hidden', type=int, default=6)
    ap.add_argument('--epochs', type=int, default=732) # Found best epoch count through manual testing
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=170) # Gemini added seed flag so weight init and splits stay consistent
    ap.add_argument('--out', default=os.path.join(outputs_dir, 'models', 'solar_model.json'))
    a = ap.parse_args()
    
    random.seed(a.seed) # seeds global random used inside NeuralNetwork weight init
    rng = random.Random(a.seed)

    pos_map = load_folder(a.pos)
    neg_map = load_folder(a.neg)
    print(f'{len(pos_map)} farm cells, {len(neg_map)} negatives (real non-farm cells)')

    # Preserve cell_id alongside features and label for out-of-fold export
    data = [(cid, x, 1) for cid, x in pos_map.items()] + [(cid, x, 0) for cid, x in neg_map.items()]
    rng.shuffle(data)
    idx = list(range(len(data)))

    posIdx = [i for i in idx if data[i][2] == 1]
    negIdx = [i for i in idx if data[i][2] == 0]
    folds = [posIdx[i::a.folds] + negIdx[i::a.folds] for i in range(a.folds)]
    foldAucs, foldAccs = [], []
    oof_rows = []

    for fold_num, f in enumerate(folds):
        fs = set(f)
        tr = [data[i] for i in idx if i not in fs]
        te = [data[i] for i in f]

        lo, hi = make_scaler([x for _, x, _ in tr])
        xTr = [scale(x, lo, hi) for _, x, _ in tr]
        tTr = [[l] for _, _, l in tr]

        net = NeuralNetwork(len(FEATURES), a.hidden, 1, direct_connections=True)
        net.fit(X=xTr, T=tTr, epochs=a.epochs)

        s = [net.predict(scale(x, lo, hi))[0] for _, x, _ in te]
        l = [lab for _, _, lab in te]

        for (cid, _, lab), score in zip(te, s):
            oof_rows.append({
                'cell_id': cid,
                'fold': fold_num,
                'actual': lab,
                'pred_score': f'{score:.6f}',
                'pred_label': int(score > 0.5)
            })

        foldAucs.append(auc(s, l))
        foldAccs.append(sum((si > 0.5) == bool(li) for si, li in zip(s, l)) / len(l))

    auc_mean, auc_std = mean(foldAucs), std(foldAucs)
    acc_mean, acc_std = mean(foldAccs), std(foldAccs)
    print(f'{a.folds}-fold CV: AUC {auc_mean:.3f} ± {auc_std:.3f} accuracy {acc_mean:.3f} ± {acc_std:.3f}')

    # Save CV summary & out-of-fold predictions to Outputs/metrics/
    metrics_dir = os.path.join(outputs_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    summary_path = os.path.join(metrics_dir, 'nn_cv_summary.csv')
    with open(summary_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['model', 'params', 'auc_mean', 'auc_std', 'acc_mean', 'acc_std'])
        writer.writeheader()
        writer.writerow({
            'model': 'NeuralNetwork',
            'params': f'hidden={a.hidden},epochs={a.epochs},direct=True',
            'auc_mean': f'{auc_mean:.6f}',
            'auc_std': f'{auc_std:.6f}',
            'acc_mean': f'{acc_mean:.6f}',
            'acc_std': f'{acc_std:.6f}'
        })

    oof_path = os.path.join(metrics_dir, 'nn_oof_predictions.csv')
    with open(oof_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['cell_id', 'fold', 'actual', 'pred_score', 'pred_label'])
        writer.writeheader()
        writer.writerows(oof_rows)

    print(f'saved {summary_path}')
    print(f'saved {oof_path}')

    # Train final model on all data
    lo, hi = make_scaler([x for _, x, _ in data])
    X = [scale(x, lo, hi) for _, x, _ in data]
    T = [[l] for _, _, l in data]
    
    net = NeuralNetwork(len(FEATURES), a.hidden, 1, direct_connections=True)
    net.fit(X=X, T=T, epochs=a.epochs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    net.save(a.out, extra={'features': FEATURES, 'lo': lo, 'hi': hi})
    print(f'saved {a.out}')


if __name__ == '__main__': train()


