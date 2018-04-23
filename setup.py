from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pygtm',
    version='0.0.2',
    description='A python implementation of Generative Topographic Mapping.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Amane Suzuki',
    author_email='amane.suzu@gmail.com',
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    url='https://github.com/amaotone/pygtm',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
