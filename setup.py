from setuptools import setup, find_packages

__version__ = '0.0.1'

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='ocatari_wrappers',
    version=__version__,
    author='Cedric Derstroff',
    author_email='cedric.derstroff@tu-darmstadt.de',
    packages=find_packages(),
    # package_data={'': extra_files},
    include_package_data=True,
    # package_dir={'':'src'},
    url='https://github.com/VanillaWhey/OCAtariWrappers',
    description='Wrappers for producing object-centric, visually masked input representations with OCAtari.',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
