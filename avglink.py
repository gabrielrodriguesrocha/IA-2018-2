import pandas as pd
import numpy as np
import unionfind as uf
import itertools
import seaborn as sns
import sys
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import squareform, pdist
from scipy.spatial import distance_matrix

def find(q, points):
    s = q
    x = points.loc[q]
    while x['parent'] != q:
        r = q
        q = x['parent']
        points.loc[r, 'parent'] = q
        x = points.loc[q]
    points.loc[s, 'cluster'] = points.loc[q, 'cluster']
    return q

def union(p, q, points):
    s = find(q, points)
    points.loc[s, 'parent'] = p

def avglink(file, kMin, kMax):
    df=pd.read_csv(file, sep='\t',header=0)
    real_clusters = pd.read_csv(file.replace('.txt', 'Real.clu'), sep='\t',header=None)
    points = df.iloc[:, :3]
    points['cluster'] = np.arange(points.shape[0])
    points['n'] = 1
    points['parent'] = np.arange(points.shape[0])
    clusters = points[['sample_label', 'cluster']]
    k = points.shape[0]
    dists = pd.DataFrame(squareform(pdist(df.iloc[:, 1:3])), columns=clusters.cluster.unique(), index=clusters.cluster.unique())
    for i in range(len(dists)): 
        dists.iat[i, i] = np.nan


    for i in range(0, points.shape[0] - 1):
        if k <= int(kMax) and k >= int(kMin):
            [find(x, points) for x in np.arange(points.shape[0])]
            print ('K = %d' % k)
            l_dict = dict(zip(set(points['cluster']), range(len(points['cluster']))))
            points = points.assign(normalized_cluster = [l_dict[x] for x in points['cluster']])
            print('ARI (Hubert-Arabie) = %lf' % adjusted_rand_score(real_clusters.values[:, 1], points['normalized_cluster'].values))
            points[['sample_label', 'normalized_cluster']].to_csv(file.split('.txt')[0] + '-avglink' + str(k) + '.clu', sep='\t', index=False, header=None)
            figname = 'plots/avglink/' + file.split('.txt')[0].split('/')[1] + '-k-' + str(k) + '.png'
            sns.pairplot(x_vars=["d1"], y_vars=["d2"], data=points, hue="normalized_cluster", height=5).savefig(figname)
        k = k - 1

        p = dists.min().idxmin()
        q = dists.idxmin().loc[p]
        r = min(p,q)
        s = max(p,q)
        dists[r] = (points.loc[r,'n'] * dists[r] + points.loc[s, 'n'] * dists[s]) / (points.loc[r,'n'] + points.loc[s,'n'])
        dists.loc[r] = (points.loc[r,'n'] * dists.loc[r] + points.loc[s, 'n'] * dists.loc[s]) / (points.loc[r,'n'] + points.loc[s,'n'])
        union(r, s, points)
        points.loc[r, 'n'] = points.loc[r, 'n'] + points.loc[s, 'n']
        dists = dists.drop(s)
        dists = dists.drop(s, axis=1)
        dists.loc[r,r] = np.nan

    return points

if __name__ == "__main__":
    avglink(sys.argv[1], sys.argv[2], sys.argv[3])