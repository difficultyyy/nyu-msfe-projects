# Asian Option Pricing

This project implements pricing of arithmetic Asian call options under the Black-Scholes framework using Monte Carlo simulation and a control variate technique.

## Methods

- Monte Carlo simulation of asset paths (GBM)
- Arithmetic average payoff (target)
- Geometric average payoff (control variate)
- Variance reduction using control variates

## Structure

- `AsianOptionBase`
  - Path simulation
  - Arithmetic and geometric payoff

- `MonteCarloPricer`
  - Standard Monte Carlo pricing

- `ControlVariatePricer`
  - Control variate method using geometric Asian option

## Tech

- Python
- NumPy
- SciPy
