from setuptools import setup

setup(
    name="TensorCraft",
    version="0.1a",
    description="High level constructs for building machine learning models with Tensorflow.",
    author='BT',
    author_email="bhaskar.tripathi@gmail.com",
    packages=["TensorCraft"],
    install_requires=["tensorflow", "h5py"],
    zip_safe=False
)
