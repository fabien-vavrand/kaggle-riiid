from setuptools import setup


def parse_requirements(file):
    with open(file) as fp:
        _requires = fp.read()
    return [e.strip() for e in _requires.split('\n') if len(e)]


setup(
    name='riiid',
    packages=['riiid'],
    version='0.0.1',
    setup_requires=['setuptools', 'wheel'],
    install_requires=parse_requirements('requirements.txt'),
)
