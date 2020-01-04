import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='berliner',
    version='0.0.10',
    author='Bo Zhang',
    author_email='bozhang@nao.cas.cn',
    description='Tools for stellar tracks & isochrones.',  # short description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/hypergravity/berliner',
    packages=setuptools.find_packages(),
    license='MIT',
    classifiers=["Development Status :: 5 - Production/Stable",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
                 "Programming Language :: Python :: 3.7",
                 "Topic :: Scientific/Engineering :: Physics",
                 "Topic :: Scientific/Engineering :: Astronomy"],
    package_dir={'berliner': 'berliner',
                 'ezpadova': 'ezpadova'},
    package_data={'ezpadova': ['*.json']},
    include_package_data=True,
    requires=['numpy', 'scipy', 'emcee', 'regli', 'joblib']
)
