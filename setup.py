from distutils.core import setup

setup(
    name='gpuarray',
    version='0.1.0',
    url='github.com/ucb-sejits/snowflake-opencl',
    license='B',
    author='Nathan Zhang',
    author_email='nzhang32@berkeley.edu',
    description="Numpy-GPU array mirroring",

    packages=[
        'gpuarray'
    ],

    install_requires=[
        'ctree',
        'numpy',
        'pycl',
    ]
)
