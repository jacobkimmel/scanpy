import pytest
import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix

import scanpy as sc
from anndata.tests.helpers import assert_equal

X_total = [[1, 0], [3, 0], [5, 6]]
X_frac = [[1, 0, 1], [3, 0, 1], [5, 6, 1]]


@pytest.mark.parametrize('typ', [np.array, csr_matrix], ids=lambda x: x.__name__)
@pytest.mark.parametrize('dtype', ['float32', 'int64'])
def test_normalize_total(typ, dtype):
    adata = AnnData(typ(X_total), dtype=dtype)
    sc.pp.normalize_total(adata, key_added='n_counts')
    assert np.allclose(np.ravel(adata.X.sum(axis=1)), [3.0, 3.0, 3.0])
    sc.pp.normalize_total(adata, target_sum=1, key_added='n_counts2')
    assert np.allclose(np.ravel(adata.X.sum(axis=1)), [1.0, 1.0, 1.0])

    adata = AnnData(typ(X_frac, dtype=dtype))
    sc.pp.normalize_total(adata, exclude_highly_expressed=True, max_fraction=0.7)
    assert np.allclose(np.ravel(adata.X[:, 1:3].sum(axis=1)), [1.0, 1.0, 1.0])


@pytest.mark.parametrize('typ', [np.array, csr_matrix], ids=lambda x: x.__name__)
@pytest.mark.parametrize('dtype', ['float32', 'int64'])
def test_normalize_total_layers(typ, dtype):
    adata = AnnData(typ(X_total), dtype=dtype)
    adata.layers["layer"] = adata.X.copy()
    sc.pp.normalize_total(adata, layers=["layer"])
    assert np.allclose(adata.layers["layer"].sum(axis=1), [3.0, 3.0, 3.0])


@pytest.mark.parametrize('typ', [np.array, csr_matrix], ids=lambda x: x.__name__)
@pytest.mark.parametrize('dtype', ['float32', 'int64'])
def test_normalize_total_view(typ, dtype):
    adata = AnnData(typ(X_total), dtype=dtype)
    v = adata[:, :]

    sc.pp.normalize_total(v)
    sc.pp.normalize_total(adata)

    assert not v.is_view
    assert_equal(adata, v)


@pytest.mark.parametrize('typ', [np.array, csr_matrix], ids=lambda x: x.__name__)
@pytest.mark.parametrize('dtype', ['float32', 'int64'])
def test_normalize_pearson_residuals(typ, dtype):
    adata = AnnData(typ(X_total), dtype=dtype)
    sc.pp.normalize_pearson_residuals(adata, theta=100., inplace=True, key_added='n_counts')
    assert (adata.X[:2, 0].mean() > adata.X[2, 0])
    assert np.allclose(adata.obs['n_counts'], [1.0, 3.0, 11.0])
    
    # test that theta influences residuals
    adata = AnnData(typ(X_total), dtype=dtype)
    x0 = sc.pp.normalize_pearson_residuals(adata, theta=100., inplace=False,)
    x1 = sc.pp.normalize_pearson_residuals(adata, theta=1., inplace=False,)
    assert np.all(x0['X'] != x1['X']) # no residuals should be equal with different theta
    

@pytest.mark.parametrize('typ', [np.array, csr_matrix], ids=lambda x: x.__name__)
@pytest.mark.parametrize('dtype', ['float32', 'int64'])
def test_normalize_pearson_residuals_layers(typ, dtype):
    adata = AnnData(typ(X_total), dtype=dtype)
    adata.layers["layer"] = adata.X.copy()
    sc.pp.normalize_pearson_residuals(adata, layers=["layer"])
    assert (adata.layers["layer"][:2, 0].mean() > adata.layers["layer"][2, 0])
    

@pytest.mark.parametrize('typ', [np.array, csr_matrix], ids=lambda x: x.__name__)
@pytest.mark.parametrize('dtype', ['float32', 'int64'])
def test_normalize_pearson_residuals(typ, dtype):
    adata = AnnData(typ(X_total), dtype=dtype)
    v = adata[:, :]

    sc.pp.normalize_pearson_residuals(v)
    sc.pp.normalize_pearson_residuals(adata)

    assert not v.is_view
    assert_equal(adata, v)
