import numpy as np


class HybridSeasonalKernelRidge:
    """Hybrid Seasonal Kernel Ridge (numpy 직접 구현, 게이트 포함).

    pkl 언피클 시 이 클래스 정의가 같은 모듈명(hskr_model)으로 import 되어 있어야 함.
    predict 시그니처가 sklearn과 다름: predict(X, months, t).
    """

    def __init__(self, n_harmonics=2, lam_season=1.0, lam_kernel=1.0, gamma=0.1, w=1.0, gate=True):
        self.nh, self.ls, self.lk, self.g, self.w, self.gate = n_harmonics, lam_season, lam_kernel, gamma, w, gate

    def _basis(self, mo, t, Xs):
        c = [np.ones_like(mo, float), (t - self._tm) / self._ts]
        for h in range(1, self.nh + 1):
            c += [np.sin(2 * np.pi * h * mo / 12.0), np.cos(2 * np.pi * h * mo / 12.0)]
        return np.column_stack([np.column_stack(c), Xs])

    def _rbf(self, A, B):
        sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A.dot(B.T)
        return np.exp(-self.g * np.maximum(sq, 0))

    def fit(self, X, y, mo, t):
        X = np.asarray(X, float); y = np.asarray(y, float).ravel()
        mo = np.asarray(mo, float); t = np.asarray(t, float)
        self._mu = X.mean(0); self._sd = X.std(0) + 1e-8
        self._tm = float(t.mean()); self._ts = float(t.std() + 1e-8)
        Xs = (X - self._mu) / self._sd
        S = self._basis(mo, t, Xs); I = np.eye(S.shape[1]); I[0, 0] = 0.0
        self.beta = np.linalg.solve(S.T @ S + self.ls * I, S.T @ y)
        r = y - S @ self.beta
        self._Xt = Xs; K = self._rbf(Xs, Xs)
        self.alpha = np.linalg.solve(K + self.lk * np.eye(len(Xs)), r); return self

    def _parts(self, X, mo, t):
        Xs = (np.asarray(X, float) - self._mu) / self._sd
        S = self._basis(np.asarray(mo, float), np.asarray(t, float), Xs)
        Kx = self._rbf(Xs, self._Xt); ker = Kx @ self.alpha
        if self.gate:
            ker = ker * Kx.max(1)
        return S @ self.beta, ker

    def predict(self, X, mo, t):
        lin, ker = self._parts(X, mo, t); return lin + self.w * ker
