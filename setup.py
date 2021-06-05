from setuptools import setup

REQUIRED_PACKAGES = ['torchvision==0.7.0',
                     'torch @ https://download.pytorch.org/whl/cpu/torch-1.6.0%2Bcpu-cp37-cp37m-linux_x86_64.whl']

setup(
    name="mivos_s2m",
    version="0.1",
    include_package_data=True,
    # scripts=["model_prediction.py", "./model/network.py"],
    packages=["model", "util", "dataset"],
    scripts=["model_prediction.py"],
    install_requires=REQUIRED_PACKAGES
)
