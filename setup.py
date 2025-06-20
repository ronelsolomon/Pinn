from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pinn',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Physics-Informed Neural Networks (PINNs) for solving differential equations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/pinn',
    packages=find_packages(),
    install_requires=[
        'torch>=1.8.0',
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
        'scipy>=1.5.0',
        'tqdm>=4.50.0',
        'scikit-learn>=0.24.0',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)
