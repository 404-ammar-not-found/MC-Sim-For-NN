# Neural Network Surrogate for Monte Carlo Option Pricing

## Overview
This project builds a neural network surrogate to approximate European call option prices generated via Monte Carlo simulation under the Black–Scholes model. The goal is to demonstrate accuracy and significant speedup compared to Monte Carlo pricing at inference time.

## What It Does
- Implements:
  - Black–Scholes analytical call price
  - Monte Carlo call option pricing
- Demonstrates Monte Carlo convergence to the Black–Scholes price
- Generates a large synthetic dataset using Monte Carlo simulation
- Trains a neural network to learn the pricing map  
  \[
  (S_0, K, T, r, \sigma) \rightarrow \text{Call Price}
  \]
- Compares:
  - Neural network vs Black–Scholes accuracy
  - Neural network vs Monte Carlo runtime performance

## Model
- Input: 5 parameters (spot, strike, maturity, rate, volatility)
- Architecture: Fully connected MLP (5 → 64 → 64 → 1)
- Loss: Mean Squared Error
- Optimizer: Adam

## Key Results
- Neural network prices closely match Black–Scholes prices
- Inference is ~80× faster than Monte Carlo with 100k simulations
- Variance in NN inference time is negligible

## Dependencies
- Python
- NumPy
- SciPy
- Matplotlib
- PyTorch

## Use Case
Fast option pricing via a learned surrogate model when repeated Monte Carlo evaluation is too slow.
