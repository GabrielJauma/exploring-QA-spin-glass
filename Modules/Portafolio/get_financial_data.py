import numpy as np
import pandas as pd
from pypfopt import risk_models, expected_returns, objective_functions

def get_returns_and_correlations(
    datafile,
    frequency=21,
    date='2016-01-01',
    ix_assets=np.arange(20)#None
):
    """Return expected returns and covariance matrix from financial data.

    Arguments
    ---------
    datafile: pandas DataFile
        Datafile with daily closing prices data.

    frequency: int
        Number of days until expected returns are met. Example: 21 for
        expected monthly returns and 252 for expected yearly returns.

    date: str or numpy DateTime64
        Historical data used to compute historical expected returns and
        correlations is taken from the beginning of time until 'date'.

    ix_assets: None or np.1darray of int
        If None then every asset in the dataset is taken. If 1darray
        then only assets with index in ix_assets are considered.

    """
    # Get data from csv.
    df = pd.read_csv(datafile, index_col=0, parse_dates=True)
    # Adjust date to dates in index. This is done to prevent errors like, e.g.
    # if 'date' is a weekend day or if markets were closed on that day.
    date = df.index[np.where(date >= df.index)[0][-1]]
    # Make historical dataframe with data until 'date'.
    hdf = df.loc[:date]
    tickers = df.columns

    # Make reduced dataframes if a size smaller than asset universe is given.
    if ix_assets is None:
        ix_assets = np.arange(df.shape[1])
    tickers = tickers[ix_assets]
    hdf = hdf[tickers]
    df = df[tickers]

    # Compute expected returns and sample covariance using historical
    # prices dataframe.
    µ = expected_returns.mean_historical_return(hdf, frequency=frequency)
    S = risk_models.sample_cov(hdf, frequency=frequency)
    # Rewrite matrices as numpy ndarrays.
    µ_mat = µ.to_numpy()
    S_mat = S.to_numpy()

    # Get asset prices.
    prices_df = df.loc[pd.Timestamp(date)]
    prices = np.ascontiguousarray(prices_df.to_numpy(), dtype=np.float64)

    return S, S_mat, µ, µ_mat, prices_df, prices


def make_QUBO(µ, S, λ, λN, prices, P0, nmin, Nq):
    """Make QUBO matrices.

    Arguments
    ---------
    µ: np.1darray of float, shape: (L,)
        Expected returns.

    S: np.2darray of float, shape: (L, L)
        Historical correlations matrix.
    λ: float
        Risk factor.

    λN: float
        Budget constraint strength.

    prices: np.1darray of float, shape: (L,)
        Closing prices of assets.

    P0: float
        Available budget.

    nmin: 1darray of int, shape: (L,)
        Minimum number of allocations allowed for each asset.

    Nq: 1darray of int, shape: (L,)
        Number of qubits used for each asset. Defines the maximum
        allowed allocation for each asset:
        nmin[i] <= n[i] <= nmin[i] + 2**Nq[i] - 1.

    Return
    ------
    muQ: np.1darray of float, shape: (dim,)
        Linear terms of the QUBO cost function.

    Q: np.1darray of float, shape: (dim, dim)
        Quadratic terms of the QUBO cost function.

    c: float
        Constant term of the QUBO cost function.

    """
    # Total number of binary variables.
    dim = int(np.sum(Nq))
    # Total number of assets.
    L = µ.size
    # Position at which the binary variables of asset i start.
    start_i = np.insert(np.cumsum(Nq[:-1]), 0, 0).astype(np.int64)

    # Weight of an individual asset.
    wia = prices/P0

    # Make sure Nq is casted as integer.
    Nq = Nq.astype(np.int64)

    # Make QUBO matrices.
    muQ = np.zeros((dim,))
    Q = np.zeros((dim, dim))

    for i in range(L):
        for r in range(Nq[i]):
            muQ[start_i[i] + r] = (
                µ[i]
                - λ*S[i]@(wia*nmin)
                - 2*λN*(wia@nmin - 1)
            )*wia[i]*np.exp2(r)

    for i in range(L):
        for j in range(L):
            for r in range(Nq[i]):
                for s in range(Nq[j]):
                    Q[start_i[i] + r, start_i[j] + s] = (
                        -wia[i]*wia[j]*(λ/2*S[i, j] + λN)*np.exp2(r+s)
                    )

    c = (
        µ@(wia*nmin)
        - λ/2*(wia*nmin)@S@(wia*nmin)
        - λN*(wia@nmin - 1)**2
    )

    return muQ, Q, c


if __name__ == '__main__':
    # Get expected returns and correlations from past data.
    _, S, _, µ, _, prices = get_returns_and_correlations(
        'Portafolio/SP500_data_from_2008-01-01.csv',
        date='2016-01-01'
    )
    
    # Number of assets.
    L = µ.size
    # Risk aversion factor.
    λ = 50
    # Budget constraint strength.
    λN = 0
    # Total budget.
    P0 = 1e5
    # Minimum allocation per asset.
    nmin = np.zeros(L)
    # Number of qubits per asset.
    Nq = np.full((L,), 3)
    
    # Make QUBO matrices.
    muQ, Q, c = make_QUBO(µ, S, λ, λN, prices, P0, nmin, Nq)
