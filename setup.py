__author__ = 'junz'

from setuptools import setup, find_packages
import io
import os

here = os.path.abspath(os.path.dirname(__file__))

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.md')

print long_description

def prepend_find_packages(*roots):
    '''
    Recursively traverse nested packages under the root directories
    '''
    packages = []

    for root in roots:
        packages += [root]
        packages += [root + '.' + s for s in find_packages(root)]

    return packages

setup(
      name='retinotopic_mapping',
      version = '1.0.0',
      url='https://github.com/zhuangjun1981/retinotopic_mapping',
      author='Jun Zhuang @ Allen Institute for Brain Science',
      install_requires=['numpy', 'scipy', 'opencv-python', 'scikit-image', 'tifffile'],
      author_email='junz@alleninstitute.org',
      description='retinotopic mapping tools',
      long_description=long_description,
      packages=prepend_find_packages('retinotopic_mapping'),
      include_package_data=True,
      package_data={'':['*.md']},
      platforms='any',
      classifiers = [
                     'Programming Language :: Python',
                     'Development Status :: 4 - Beta',
                     'Natural Language :: English',
                     'Operating System :: OS Independent',
                     ]
       )
