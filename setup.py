from setuptools import setup

setup(
    name='ts_forecast',
    url='https://github.com/alexbotsula/package_ts_forecast',
    author='Alex Botsula',
    author_email='abotsula@gmail.com',
    packages=['ts_forecast'],
    install_requires=['pandas', 'numpy', 'scikit-learn', 'datetime', 'matplotlib', 'statsmodels', 'keras'],
    license='MIT',
    description='Package includes utility classes and function for time series transformations and forecasting'
)