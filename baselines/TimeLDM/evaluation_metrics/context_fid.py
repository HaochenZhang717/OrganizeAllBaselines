import sys
import os
import scipy
import numpy as np

# Add the project root to sys.path so the shared ts2vec package is importable.
# This file is at baselines/TimeLDM/evaluation_metrics/context_fid.py,
# so going up 3 directories from here reaches the project root.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..')
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation_metrics.ts2vec.ts2vec import TS2Vec


def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def Context_FID(ori_data, generated_data):
    model = TS2Vec(
        input_dims=ori_data.shape[-1],
        device=0,
        batch_size=8,
        lr=0.001,
        output_dims=320,
        max_train_length=3000,
    )
    model.fit(ori_data, verbose=False)
    ori_repr = model.encode(ori_data, encoding_window='full_series')
    gen_repr = model.encode(generated_data, encoding_window='full_series')
    idx = np.random.permutation(ori_data.shape[0])
    ori_repr = ori_repr[idx]
    gen_repr = gen_repr[idx]
    return calculate_fid(ori_repr, gen_repr)
