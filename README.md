# VIX_vertex_prediction

### CodeBase
- a directory including all files used for this research
- nb0_*.ipynb : are notebook files revolving data extraction, cleaning, target creation, and feature engineering. The notebooks include visuals of the process.
- s1_*.py : files revolve arround the ETL-pipeline, creating functions for used to load data in the project
- s2_*.py : files are used for creating and training LSTM models, along with the baseline model
- s3_*.py : The files numbering are in chronological order and revolves evaluations and visualizations of predictions from the ML models.
Note: Some files still remain in the directory as they can be used for future work, in terms of using VIX options premiums to calculate the returns. The directory "CodeBase\Data\Options_price_data" includes historical listings, and synthetically estimated option premiums in a SQLite database, created using the black scholes model, and inferred volatility. 


### Abstract:
This study investigates the use of Long ShortTerm Memory (LSTM) neural networks for predicting the Volatility Index (VIX), focusing on
two main aspects: the comparative information
gain of transformed technical indicators versus
Open-High-Low-Close (OHLC) data. Additionally, we seek to find the efficacy of a custom
loss function specifically designed to weigh predictions of high target values. As part of this
research, we also develop a novel target variable, aimed at defining a floating points values
to describe a local price ceiling or floor, relative to the current price. We evaluate model
performance through loss scores and trading
simulations, emphasizing cumulative and average returns. As all elements of the project
work in symbiosis, we dubbed it ”VIX Vertex
Predictions”. While the VIX is not a trade-able
index, this research aims to show the predictive
capability of the different input and model configurations. Throughout the report, we strongly
emphasize that the trading simulator is merely a
tool for simplifying evaluation. Findings reveal
that technical indicators initially worsened the
loss score’s compared to models using OHLC
data. While running trading simulations, the
combination of technical indicators and the custom loss function outperformed other combinations on the test set, however, the mixed results
in the of the combinations can only suggests
a limited yet potential information gain from
technical indicators when paired with the custom loss function. The study aims to contribute
to the field of financial forecasting with the
use of time series neural networks, while highlighting future research avenues for enhancing
predictive accuracy for the volatility index.
