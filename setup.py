__author__ = 'junz'

import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import io
import os

here = os.path.abspath(os.path.dirname(__file__))
os.chdir(here)

package_root = 'retinotopic_mapping'

# get install requirements
with open('requirements.txt') as req_f:
    install_reqs = req_f.read().splitlines()
install_reqs = [ir for ir in install_reqs if '#' not in ir]
install_reqs = install_reqs[::-1]
print('\ninstall requirements:')
print('\n'.join(install_reqs))
print('')

# get long_description
def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)
long_description = read('README.md')

# Recursively traverse nested packages under the root directories
# packages = [f[0] for f in os.walk(package_root)]
# packages = [f.replace('\\', '/') for f in packages]
# packages = [unicode(f) for f in packages if f[-18:] != '.ipynb_checkpoints']

packages = [package_root]
packages += [package_root + '.' + s for s in find_packages(package_root)]
packages += ['{}/examples'.format(package_root)]
packages += ['{}/test'.format(package_root)]
packages += ['{}/other_test'.format(package_root)]
print('\npackages to be installed:')
print('\n'.join(packages))

# define tests
class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['--junitxml=result.xml']
        self.test_args_cov = self.test_args + ['--cov=retinotopic_mapping', '--cov-report=term', '--cov-report=html']
        self.test_suite = True

    def run_test(self):
        import pytest

        try:
            errcode = pytest.main(self.test_args_cov)
        except:
            errcode = pytest.main(self.test_args)
        sys.exit(errcode)


# setup
setup(
      name='retinotopic_mapping',
      version = '2.5.0',
      url='https://github.com/zhuangjun1981/retinotopic_mapping',
      author='Jun Zhuang @ Allen Institute for Brain Science',
      install_requires=install_reqs,
      cmdclass={'test': PyTest},
      author_email='junz@alleninstitute.org',
      description='retinotopic mapping tools',
      long_description=long_description,
      packages=packages,
      include_package_data=True,
      package_data={'':['*.md']},
      platforms='any',
      classifiers=['Programming Language :: Python',
                   'Development Status :: 4 - Beta',
                   'Natural Language :: English',
                   'Operating System :: OS Independent',],
      )
