from setuptools import setup, find_packages

setup(name='gym_hybrid',
      version='0.0.1',
      packages=find_packages(),  # Automatically finds the package folders
      install_requires=['gymnasium>=0.29.1', 'numpy', 'pygame'],
)
