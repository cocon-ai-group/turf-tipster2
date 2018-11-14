import os
import time
import pickle
import sys
import gc
import numpy as np
import pandas as pd

print('read data:')

X = pd.read_csv('horse_history.csv', index_col=0, header=None)

import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

fig, ax = plt.subplots(ncols=5,nrows=4,figsize=(20,16),sharex='row',sharey='row')
print('PCA')
time_fs = time.time()
pca = PCA(n_components=10)
pca_X = pd.DataFrame(pca.fit_transform(X))
for i in range(0,10,2):
	pca_X.plot(kind='scatter',x=i,y=i+1,s=8,ax=ax[0,i//2])
ax[0,0].set_title('PCA')
print('%f sec.'%(time.time() - time_fs))

print('TruncatedSVD')
time_fs = time.time()
tsvd = TruncatedSVD(n_components=10)
tsvd_X = pd.DataFrame(tsvd.fit_transform(X))
for i in range(0,10,2):
	tsvd_X.plot(kind='scatter',x=i,y=i+1,s=8,ax=ax[1,i//2])
ax[1,0].set_title('TruncatedSVD')
print('%f sec.'%(time.time() - time_fs))

print('GaussianRandomProjection')
time_fs = time.time()
grp = GaussianRandomProjection(n_components=10, eps=0.1)
grp_X = pd.DataFrame(grp.fit_transform(X))
for i in range(0,10,2):
	grp_X.plot(kind='scatter',x=i,y=i+1,s=8,ax=ax[2,i//2])
ax[2,0].set_title('GaussianRandomProjection')
print('%f sec.'%(time.time() - time_fs))

print('SparseRandomProjection')
time_fs = time.time()
srp = SparseRandomProjection(n_components=10, dense_output=True)
srp_X = pd.DataFrame(srp.fit_transform(X))
for i in range(0,10,2):
	srp_X.plot(kind='scatter',x=i,y=i+1,s=8,ax=ax[3,i//2])
ax[3,0].set_title('SparseRandomProjection')
print('%f sec.'%(time.time() - time_fs))

fig.savefig('manifold1.png')
plt.clf()
plt.close()

fig, ax = plt.subplots(ncols=5,nrows=4,figsize=(20,16),sharex=True,sharey=True)
for i in range(0,10,2):
	grp_X.plot(kind='scatter',x=i,y=i+1,s=8,ax=ax[0,i//2],c=X[16],cmap='gnuplot',colorbar=(i==8)) # レース数
ax[0,0].set_title('Num of race')
for i in range(0,10,2):
	grp_X.plot(kind='scatter',x=i,y=i+1,s=8,ax=ax[1,i//2],c=X[17],cmap='gnuplot',colorbar=(i==8)) # 平均着順
ax[1,0].set_title('Ave. rank')
for i in range(0,10,2):
	grp_X.plot(kind='scatter',x=i,y=i+1,s=8,ax=ax[2,i//2],c=X[22],cmap='gnuplot',colorbar=(i==8)) # 平均人気
ax[2,0].set_title('Ave. favor')
for i in range(0,10,2):
	grp_X.plot(kind='scatter',x=i,y=i+1,s=8,ax=ax[3,i//2],c=X[23],cmap='gnuplot',colorbar=(i==8)) # 平均単勝オッズ
ax[3,0].set_title('Ave. odds')

fig.savefig('manifold2.png')
plt.clf()
plt.close()

print('end')
