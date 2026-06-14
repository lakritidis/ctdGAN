import os
import sys

import numpy as np
import pandas as pd
import time
from ctdgan import ctdGAN

num_threads = 1
os.environ['OMP_NUM_THREADS'] = str(num_threads)
np.set_printoptions(linewidth=400, threshold=sys.maxsize)
seed = 1

dataset_path = '/media/leo/7CE54B377BB9B18B/datasets/Imbalanced/bin_mixed/heart.csv'
categorical_columns = (1, 2, 6, 8, 10, 12)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    random_state = 42

    df = pd.read_csv(dataset_path)
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    ctdgan = ctdGAN(embedding_dim=128, discriminator=(256, 256), generator=(256, 256), epochs=300, batch_size=100,
                    pac=10, max_clusters=20, cluster_method='kmeans', scaler='mms11', use_classifier=True,
                    sampling_strategy='auto', alpha_k=0.07, random_state=random_state)

    t_s = time.time()
    balanced_data = ctdgan.fit_resample(x, y, categorical_columns=categorical_columns)

    print("Balanced Data shape:", balanced_data[0].shape)
    print(balanced_data[0])
    print("Finished in", time.time() - t_s, "sec")
