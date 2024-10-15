from distutils.core import setup

setup(
    name="deep_hedging",
    packages=[
        "deep_hedging",
        "deep_hedging/base",
        "deep_hedging/config",
        "deep_hedging/curve",
        "deep_hedging/dl",
        "deep_hedging/dl/baselines",
        "deep_hedging/dl/models",
        "deep_hedging/fixed_income",
        "deep_hedging/hedger",
        "deep_hedging/linear",
        "deep_hedging/monte_carlo",
        "deep_hedging/monte_carlo/bid_ask",
        "deep_hedging/monte_carlo/rates",
        "deep_hedging/monte_carlo/spot",
        "deep_hedging/monte_carlo/volatility",
        "deep_hedging/non_linear",
        "deep_hedging/non_linear/exotic",
        "deep_hedging/non_linear/exotic/basket",
        "deep_hedging/non_linear/exotic/two_assets",
        "deep_hedging/non_linear/vanilla",
        "deep_hedging/rl",
        "deep_hedging/underlyings",
        "deep_hedging/utils",
    ],
    version="1.11",
    license="MIT",
    description="Hedging Derivatives Under Incomplete Markets with Deep Learning",
    long_description=open("README").read(),
    author="Viacheslav Buchkov",
    author_email="viacheslav.buchkov@gmail.com",
    url="https://github.com/v-buchkov/deep-hedging",
    download_url="https://github.com/v-buchkov/deep-hedging/archive/refs/tags/v1.8.tar.gz",
    keywords=[
        "deep-hedging",
        "deep hedging",
        "derivatives",
        "hedging",
        "deep learning",
        "reinforcement learning",
    ],
    install_requires=[
        "numpy",
        "numpy-financial",
        "pandas",
        "scikit-learn",
        "torch",
        "matplotlib",
        "tqdm",
        "IPython",
        "yfinance",
        "gym",
        "scipy"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
