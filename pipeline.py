import numpy as np
import pandas as pd
from astroquery.sdss import SDSS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import logging
import os
import sys

# Configure Logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
ABS_MAG_GALAXY = -21 
SPEED_OF_LIGHT = 299792.458  # km/s
BATCH_SIZE = 50000           # Process data in chunks to manage memory

class UniversePipeline:
    """
    Scalable ETL pipeline for cosmological data processing.
    Handles data ingestion, physics-based transformation, and regression analysis.
    """
    
    def __init__(self, output_file="processed_universe_data.csv"):
        self.output_file = output_file
        self.model = None
        self.hubble_constant = None

    def _fetch_data_generator(self, total_limit: int):
        """
        Generator function to simulate chunking from SDSS.
        This enables processing of datasets larger than available RAM.
        """
        offset = 0
        while offset < total_limit:
            current_limit = min(BATCH_SIZE, total_limit - offset)
            logger.info(f"Fetching batch: Rows {offset} to {offset + current_limit}...")

            query = f"""
            SELECT TOP {current_limit} 
                objid, 
                z as redshift, 
                petroMag_r as mag_r 
            FROM SpecPhoto 
            WHERE 
                class = 'GALAXY' 
                AND z > 0.01 AND z < 0.4 
                AND zWarning = 0 
                AND petroMag_r < 18 
            """
            try:
                res = SDSS.query_sql(query)
                if res is None:
                    break
                yield res.to_pandas()
                offset += current_limit
            except Exception as e:
                logger.error(f"Query failed at offset {offset}: {e}")
                break

    def process_etl(self, total_rows: int = 100000):
 
        logger.info("Starting ETL Pipeline...")
        
        # Initialize output file with header
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
        
        chunk_iterator = self._fetch_data_generator(total_rows)
        
        total_processed = 0

        for i, df_chunk in enumerate(chunk_iterator):
            # 1. CLEANING
            df_chunk = df_chunk.dropna()
            
            # 2. PHYSICS TRANSFORMATION (Vectorized for speed)
            # Calculate recessional velocity (v = cz)
            df_chunk['velocity_kms'] = df_chunk['redshift'] * SPEED_OF_LIGHT
            
            # Calculate distance (Distance Modulus Formula)
            # d = 10^((m - M + 5) / 5)
            df_chunk['distance_pc'] = 10 ** ((df_chunk['mag_r'] - ABS_MAG_GALAXY + 5) / 5)
            df_chunk['distance_mpc'] = df_chunk['distance_pc'] / 1e6
            
            # 3. OUTLIER REMOVAL (Local chunk filtering)
            q_low = df_chunk['distance_mpc'].quantile(0.05)
            q_high = df_chunk['distance_mpc'].quantile(0.95)
            df_clean = df_chunk[(df_chunk['distance_mpc'] < q_high) & 
                                (df_chunk['distance_mpc'] > q_low)]

            # 4. LOAD (Append to CSV)
            write_header = (i == 0)
            df_clean.to_csv(self.output_file, mode='a', header=write_header, index=False)
            
            total_processed += len(df_clean)
            logger.info(f"Batch {i+1} processed. Total records: {total_processed}")
            
            # Memory Management: Explicitly delete chunk to free RAM
            del df_chunk, df_clean

        logger.info(f"ETL Complete. Data persisted to {self.output_file}")

    def run_analysis(self):
        """
        Loads processed data and runs statistical regression.
        """
        if not os.path.exists(self.output_file):
            logger.error("No processed data found. Run ETL first.")
            return

        logger.info("Loading processed data for Analysis...")
        # Load the optimized dataset 
        df = pd.read_csv(self.output_file)
        
        X = df['distance_mpc'].values.reshape(-1, 1)
        y = df['velocity_kms'].values

        logger.info("Fitting Linear Regression Model...")
        self.model = LinearRegression(fit_intercept=False) # Hubble's law passes through origin
        self.model.fit(X, y)

        self.hubble_constant = self.model.coef_[0]
        r2 = r2_score(y, self.model.predict(X))

        logger.info(f"RESULTS: Hubble Constant = {self.hubble_constant:.2f} km/s/Mpc")
        logger.info(f"RESULTS: Model R^2 = {r2:.4f}")

        return df, X

    def visualize(self, df, X):
        logger.info("Generating Visualization...")
        plt.figure(figsize=(10, 6))
        
        plot_data = df.sample(n=min(10000, len(df)), random_state=42)
        
        sns.scatterplot(x=plot_data['distance_mpc'], y=plot_data['velocity_kms'], 
                        s=10, alpha=0.3, color='#2c3e50', label='SDSS Galaxies (Sample)')
        
        # Regression Line (Using the fitted model)
        x_range = np.linspace(df['distance_mpc'].min(), df['distance_mpc'].max(), 100).reshape(-1, 1)
        y_pred = self.model.predict(x_range)
        
        plt.plot(x_range, y_pred, color='#e74c3c', linewidth=2, 
                 label=f'Hubble Law Fit ($H_0$={self.hubble_constant:.2f})')
        
        plt.title(f"Hubble's Law: Expansion of the Universe\n(Pipeline Result: {len(df)} Objects)", fontsize=14)
        plt.xlabel("Distance (Mpc)", fontsize=12)
        plt.ylabel("Velocity (km/s)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        plt.savefig("hubble_pipeline_output.png", dpi=300)
        logger.info("Visualization saved.")
        plt.show()

if __name__ == "__main__":
    pipeline = UniversePipeline()
    pipeline.process_etl(total_rows=100000) 
    data, X_reg = pipeline.run_analysis()
    pipeline.visualize(data, X_reg)