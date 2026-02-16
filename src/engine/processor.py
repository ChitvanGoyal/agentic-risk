import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

class AmexDataProcessor:
    def __init__(self, config):
        self.config = config

    def load_and_clean(self, file_path):
        # Using a subset for development
        df = pd.read_csv(file_path, nrows=50000)
        df['S_2'] = pd.to_datetime(df['S_2'])
        # Sort by customer and date to ensure time continuity
        df = df.sort_values(['customer_ID', 'S_2'])
        # Create time index for TFT
        df['time_idx'] = df.groupby('customer_ID').cumcount()
        return df

    def create_dataset(self, df):
        return TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="target",
            group_ids=["customer_ID"],
            min_encoder_length=1,
            max_encoder_length=self.config['model']['context_length'],
            min_prediction_length=1,
            max_prediction_length=self.config['model']['prediction_length'],
            static_categoricals=["customer_ID"],
            # P_2 is a key liquidity feature in AMEX data
            time_varying_unknown_reals=["P_2", "D_39", "B_1"],
            add_relative_time_idx=True,
            add_target_history=True,
            allow_missing_timesteps=True
        )