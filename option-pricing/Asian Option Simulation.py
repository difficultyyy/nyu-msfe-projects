import numpy as np
import pandas as pd
from scipy.stats import norm


class AsianOptionBase:
    def __init__(self, r, sigma, K, S0, T, m, seed=42):
        self.r = r
        self.sigma = sigma
        self.K = K
        self.S0 = S0
        self.T = T
        self.m = m
        self.dt = T/m

        np.random.seed(seed)

    def simulate_paths(self, n):
        Z = np.random.randn(n, self.m)

        W = np.cumsum(np.sqrt(self.dt)*Z, axis=1)

        t = np.linspace(self.dt, self.T, self.m)

        S = self.S0*np.exp((self.r - 0.5*self.sigma**2)*t+self.sigma*W)
        return S

    def arithmetic_payoff(self, S):
        A = np.mean(S, axis=1)
        return np.exp(-self.r*self.T)*np.maximum(A-self.K, 0)

    def geometric_payoff(self, S):
        G = np.exp(np.mean(np.log(S), axis=1))
        return np.exp(-self.r*self.T)*np.maximum(G-self.K, 0)


class MonteCarloPricer(AsianOptionBase):

    def __init__(self, r, sigma, K, S0, T, m, seed=42):
        super().__init__(r, sigma, K, S0, T, m, seed)

    def price(self, n):
        S = self.simulate_paths(n)
        C = self.arithmetic_payoff(S)
        price = np.mean(C)
        error = np.std(C)/np.sqrt(n)
        return price, error


class ControlVariatePricer(AsianOptionBase):

    def __init__(self, r, sigma, K, S0, T, m, seed=42):
        super().__init__(r, sigma, K, S0, T, m, seed)

    def geometric_price(self):
        ti = np.linspace(self.T/self.m, self.T, self.m)
        T_bar = np.mean(ti)

        sigma_bar_sq = (self.sigma**2 / (self.m**2 * T_bar)) * \
            np.sum((2*np.arange(1, self.m+1)-1)*ti[::-1])

        sigma_bar = np.sqrt(sigma_bar_sq)
        delta = 0.5*self.sigma**2 - 0.5*sigma_bar_sq

        d1 = (np.log(self.S0/self.K) + (self.r - delta + 0.5*sigma_bar_sq)*T_bar) \
            / (sigma_bar*np.sqrt(T_bar))

        d2 = d1 - sigma_bar*np.sqrt(T_bar)

        price = np.exp(-delta*T_bar - self.r*(self.T - T_bar)) * self.S0 * norm.cdf(d1) \
            - np.exp(-self.r*self.T) * self.K * norm.cdf(d2)

        return price

    def price(self, n):
        S = self.simulate_paths(n)
        Y = self.arithmetic_payoff(S)
        X = self.geometric_payoff(S)
        EX = self.geometric_price()

        cov = np.cov(X, Y)[0, 1]
        varX = np.var(X)

        b = cov/varX

        Y_cv = Y-b*(X-EX)
        price = np.mean(Y_cv)
        error = np.std(Y_cv)/np.sqrt(n)
        corr = np.corrcoef(X, Y)[0, 1]

        return price, error, corr


mc = MonteCarloPricer(0.01, 0.3, 100, 110, 1, 12)
cv = ControlVariatePricer(0.01, 0.3, 100, 110, 1, 12)
n = 1000000

mc_price, mc_err = mc.price(n)
cv_price, cv_err, corr = cv.price(n)

print(f"N = {n}")
print("----- Monte Carlo -----")
print(f"Price: {mc_price:.6f}")
print(f"Standard Error: {mc_err:.6f}")

print("\n----- Control Variates -----")
print(f"Price: {cv_price:.6f}")
print(f"Standard Error: {cv_err:.6f}")
print(f"Correlation (X,Y): {corr:.6f}")
