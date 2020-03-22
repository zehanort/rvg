from distutils.core import setup

setup(
    name='rvg',
    version='0.1',
    description='random values generator',
    long_descrption=open('README.md', 'r').read(),
    author='Sotiris Niarchos',
    author_email='sot.niarchos@gmail.com',
    # url='',
    # download_url='',
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
    packages=['rvg']
)
