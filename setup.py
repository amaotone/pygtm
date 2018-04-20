from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pygtm',
    version='0.0.1',
    description='A python implementation of Generative Topographic Mapping.',
    long_description=readme,
    author='Amane Suzuki',
    author_email='amane.suzu@gmail.com',
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    url='https://github.com/amaotone/pygtm',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
