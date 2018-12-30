import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import squareform, pdist
from scipy.spatial import distance_matrix

def kmeans(file, max_iter, k):
    df=pd.read_csv(file, sep='\t',header=0)
    real_clusters = pd.read_csv(file.replace('.txt', 'Real.clu'), sep='\t',header=None)
    clusters = df.iloc[np.random.randint(low=0, high=df.shape[0] - 1, size=int(k)), 1:]
    points = df.iloc[:, :3]
    points['cluster'] = 0

    for i in range(int(max_iter)):
        d = pd.DataFrame(distance_matrix(points.values[:, 1:3], clusters.values[:, :2]))
        points['cluster'] = d.values.argmin(axis=1)
        clusters = points.groupby('cluster').mean()


    #print(d)
    l_dict = dict(zip(set(points['cluster']), range(len(points['cluster']))))
    points = points.assign(normalized_cluster = [l_dict[x] for x in points['cluster']])
    print('ARI (Hubert-Arabie) = %lf' % adjusted_rand_score(real_clusters.values[:, 1], points['normalized_cluster'].values))
    figname = 'plots/kmeans/' + file.split('.txt')[0].split('/')[1] + '-k-' + str(k) + '-iter-' + str(max_iter) + '.png'
    sns.pairplot(x_vars=["d1"], y_vars=["d2"], data=points, hue="normalized_cluster", height=5).savefig(figname)
    return points[['sample_label', 'normalized_cluster']].to_csv(file.split('.txt')[0] + '-kmeans.clu', sep='\t', index=False, header=None)