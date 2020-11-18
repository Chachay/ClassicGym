import setuptools

setuptools.setup(
    name="ClassicGym", 
    version="0.0.1",
    author="Chachay",
    author_email="Chachay@users.noreply.github.com",
    description="OpenAI Gym environments for classic (nonlinear) problems",
    url="https://github.com/Chachay/ClassicGym",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
      'sympy', 'scipy', 'matplotlib', 'numpy>=1.10.4', 'gym>=0.17.2'
    ],
)
