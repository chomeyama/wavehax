import os

from setuptools import find_packages, setup

from wavehax import __version__

requirements = {
    "install": [
        "wheel",
        "setuptools",
        "protobuf",
        "PyYAML",
        "tqdm",
        "h5py",
        "librosa",
        "soundfile",
        "pyloudnorm",
        "pyworld",
        "pysptk",
        "einops",
        "matplotlib",
        "hydra-core>=1.2",
        "torch>=1.9.0",
        "torchaudio>=0.8.1",
        "torchprofile",
        "transformers",
    ],
    "setup": [
        "numpy",
    ],
}

entry_points = {
    "console_scripts": [
        "wavehax-extract-features=wavehax.bin.extract_features:main",
        "wavehax-compute-statistics=wavehax.bin.compute_statistics:main",
        "wavehax-profile=wavehax.bin.profile:main",
        "wavehax-train=wavehax.bin.train:main",
        "wavehax-decode=wavehax.bin.decode:main",
    ]
}

install_requires = requirements["install"]
setup_requires = requirements["setup"]

readme_path = os.path.join(os.path.dirname(__file__), "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="wavehax",
    version=__version__,
    author="Reo Yoneyama",
    author_email="yoneyama.reo@g.sp.m.is.nagoya-u.ac.jp",
    url="http://github.com/chomeyama/wavehax",
    description="Wavehax official implementation",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=install_requires,
    setup_requires=setup_requires,
    entry_points=entry_points,
)
