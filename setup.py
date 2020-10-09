from setuptools import setup, find_packages

setup(name='SNIaDCA', version='0.1',
      description='Predict Branch groups and M_B-vs-VSi groups using GMMs.',
      url='https://github.com/anthonyburrow/SNIaDCA',
      author='Anthony Burrow',
      author_email='anthony.r.burrow-1@ou.edu',
      license='GPL-v3',
      packages=find_packages(),
      package_data={'SNIaDCA': ['models/*.p']},
      include_package_data=True,
      install_requires=['numpy', 'scikit-learn']
      )
