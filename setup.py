from setuptools import setup
setup(name='Block_matrices',
      version='1.0',
      description='Module for handling large matrices,such as those from tbtrans',
      url='',
      author='Aleksander Bach Lorentzen',
      author_email='aleksander.bl.mail@gmail.com',
      license='MIT',
      packages=['Block_matrices'],
      zip_safe=False,
      install_requires = ["numpy", 
                          "numba", 
                          "siesta_python",
                          "scipy",
                          "matplotlib"
                         ]
      )

