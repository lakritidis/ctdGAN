from distutils.core import setup
from setuptools import find_packages

DESCRIPTION = 'A Generative Adversarial Network for synthesizing artificial tabular data.'

LONG_DESCRIPTION = '<p>ctdGAN is a Conditional Generative Adversarial Network for alleviating class imbalance in '\
    'tabular datasets. The model is based on an initial space partitioning step that assigns cluster labels to the '\
    'input samples. These labels are used to synthesize samples via a probabilistic sampling mechanism. ctdGAN '\
    'optimizes a loss function that is sensitive to both cluster and class mis-predictions, rendering the model'\
    'capable of generating samples in subspaces that resemble those of the original data distribution.</p>'\
    '<p><b>Licence:</b> Apache License, 2.0 (Apache-2.0)</p>' \
    '<p><b>Dependencies:</b>NumPy, pandas, Matplotlib, seaborn, joblib, ' \
    'Reversible Data Transforms(RDT), scikit-learn, pytorch, Synthetic Data Vault</p>'\
    '<p><b>GitHub repository:</b> '\
    '<a href="https://github.com/lakritidis/ctdGAN">https://github.com/lakritidis/ctdGAN</a></p>'

setup(
    name='ctdgan',
    version='0.1.1',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author="Leonidas Akritidis",
    author_email="lakritidis@ihu.gr",
    maintainer="Leonidas Akritidis",
    maintainer_email="lakritidis@ihu.gr",
    packages=find_packages(),
    package_data={'': ['generators/*']},
    url='https://github.com/lakritidis/ctdGAN',
    install_requires=["numpy",
                      "pandas",
                      "matplotlib",
                      "seaborn",
                      "sdv",
                      "tqdm",
                      "joblib",
                      "torch>=2.0.0",
                      "scikit-learn>=1.4.0",
                      "rdt>=1.3.0,<2.0"],
    license="Apache",
    keywords=[
        "ctdGAN", "GAN", "Generative Adversarial Network", "imbalanced data", "tabular data", "deep learning"]
)
