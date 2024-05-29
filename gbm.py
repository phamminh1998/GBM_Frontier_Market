# Define function to run brownian motion simulation
# Set seed for random number generators
import random
import numpy as np

import os
import pandas as pd

class GBM_Simulator:
    """
    This callable class will generate a daily
    close price based DataFrame to simulate
    asset pricing paths with Geometric Brownian Motion (GBM) for pricing.

    It will output the results to a CSV with the ticker symbol.

    Parameters
    ----------
    start_date : `str`
        The starting date in YYYY-MM-DD format.
    end_date : `str`
        The ending date in YYYY-MM-DD format.
    output_dir : `str`
        The full path to the output directory for the CSV.
    symbol : `str`
        The ticker symbol to use.
    init_price : `float`
        The initial price of the asset.
    mu : `float`
        The mean 'drift' of the asset.
    sigma : `float`
        The 'volatility' of the asset.
    num_sims : `int`
        Number of GBM paths to simulate.
    """

    def __init__(
        self,
        start_date,
        end_date,
        output_dir,
        symbol,
        init_price,
        mu,
        sigma,
        num_sims
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        self.symbol = symbol
        self.init_price = init_price
        self.mu = mu
        self.sigma = sigma
        self.num_sims = num_sims

    def _create_empty_frame(self):
        """
        Creates the empty Pandas DataFrame with a date column using
        business days between two dates. Each of the price columns
        are set to zero.

        Returns
        -------
        `pd.DataFrame`
            The empty close price DataFrame for subsequent population.
        """
        date_range = pd.date_range(
            self.start_date,
            self.end_date,
            freq='B'
        )

        return pd.DataFrame(
            {
                'date': date_range,
                **{f'path_{i}': np.nan for i in range(1, self.num_sims + 1)}
            }
        )

    def _create_geometric_brownian_motion(self, data):
        """
        Calculates asset price paths using the analytical solution
        to the Geometric Brownian Motion stochastic differential
        equation (SDE).

        Parameters
        ----------
        data : `pd.DataFrame`
            The DataFrame needed to calculate length of the time series.

        Returns
        -------
        `pd.DataFrame`
            The DataFrame containing the asset price paths.
        """
        n = len(data)
        T = n / len(data['date'])
        dt = T / n

        paths = pd.DataFrame(
            self.init_price * np.exp(
                np.cumsum((self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * np.random.normal(0, 1, size=(n, self.num_sims)), axis=0)
            ),
            columns=[f'path_{i}' for i in range(1, self.num_sims + 1)]
        )
        paths['average'] = paths.mean(axis=1)
        return pd.concat([data[['date']], paths], axis=1)

    def _output_frame_to_dir(self, data):
        """
        Output the fully-populated DataFrame to disk into the
        desired output directory.

        Parameters
        ----------
        data : `pd.DataFrame`
            The DataFrame containing the generated price paths.
        """
        output_file = os.path.join(self.output_dir, f'{self.symbol}.csv')
        data.to_csv(output_file, index=False)

    def __call__(self):
        """
        The entrypoint for generating the asset price frame. Firstly, this
        generates an empty frame. It then populates this
        frame with some simulated GBM data and saves it to disk as a CSV.
        """
        data = self._create_empty_frame()
        paths = self._create_geometric_brownian_motion(data)
        self._output_frame_to_dir(paths)