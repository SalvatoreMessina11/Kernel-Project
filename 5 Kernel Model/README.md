# 5 Kernel Model

This stage estimates the rolling kernel models and constructs the constrained active portfolios.

Expected work includes fitting linear, polynomial, and Gaussian/RBF kernel specifications; generating next-day return forecasts; estimating training-window covariance matrices; and solving the constrained long-only portfolio problem.

Use `output` for forecasts, weights, returns, and model summaries. Use `img` for model-level diagnostics.
