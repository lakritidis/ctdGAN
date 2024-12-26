import os
import sys

import numpy as np
import time

from ctdGAN.ctd_gan import ctdGAN
from ctdGAN.TabularDataset import TabularDataset
from ctdGAN.Tools import set_random_states

num_threads = 1
os.environ['OMP_NUM_THREADS'] = str(num_threads)
np.set_printoptions(linewidth=400, threshold=sys.maxsize)
seed = 1

# dataset_path = '/media/leo/7CE54B377BB9B18B/datasets/Imbalanced/multiclass_continuous/'
dataset_path = 'D:/datasets/Imbalanced/multiclass_continuous/'
datasets = {
    'anemia': {'path': dataset_path + 'anemia.csv', 'categorical_cols': (), 'class_col': 14},
}


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    set_random_states(seed)
    dataset = datasets['anemia']

    dset = TabularDataset(name='test', class_column=dataset['class_col'],
                          categorical_columns=dataset['categorical_cols'],  random_state=seed)
    dset.load_from_csv(path=dataset['path'])
    dset.display_params()

    x = dset.x_
    y = dset.y_

    t_s = time.time()

    gan = ctdGAN(discriminator=(256, 256), generator=(256, 256), epochs=10, batch_size=100,
                 pac=10, embedding_dim=128, max_clusters=20, cluster_method='kmeans', scaler='mms11',
                 sampling_strategy='create-new', random_state=seed)

    balanced_data = gan.fit_resample(x, y, categorical_columns=dataset['categorical_cols'])

    print("Balanced Data shape:", balanced_data[0].shape)
    # print(balanced_data[0])
    print("Finished in", time.time() - t_s, "sec")
