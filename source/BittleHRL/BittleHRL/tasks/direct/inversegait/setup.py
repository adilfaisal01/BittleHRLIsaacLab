from setuptools import setup, find_packages

setup(
    name='inversegait',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',
        'scipy'
    ],
    author='Adil Faisal',
    description='A library for gait generation and inverse kinematics for the Bittle quadruped robot.',
    url='https://github.com/adilfaisal01/SE952--Bittle-Quadruped',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
