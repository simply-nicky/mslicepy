from setuptools import setup, find_packages

with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(name='mslicepy',
      version='0.0.1',
      author='Nikolay Ivanov',
      author_email="nikolay.ivanov@desy.de",
      long_description=long_description,
      long_description_content_type='text/markdown',
      url="https://github.com/simply-nicky/mslicepy",
      packages=find_packages(),
      include_package_data=True,
      package_data={'mslicepy': ['data/*.npz']},
      install_requires=['numpy'],
      extras_require={'interactive': ['matplotlib', 'jupyter', 'pyximport']},
      classifiers=[
          "Programming Language :: Python",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent"
      ],
      python_requires='>=3.6')
