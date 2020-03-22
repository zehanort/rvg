from setuptools import setup

setup(
    name='rvg',
    version='0.0.1',
    description='random values generator',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    author='Sotiris Niarchos',
    author_email='sot.niarchos@gmail.com',
    url='http://pypi.python.org/pypi/rvg',
    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities'
    ],

    install_requires=['numpy'],
    python_requires='>=3.5',
    packages=['rvg']
)
