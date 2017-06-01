from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop

import os


try:
    from grpc.tools import command
except ImportError:
    build_requires = ['grpcio-tools']
else:
    build_requires = []


class PackgeProtoBuilderMixin(object):
    def build_package_protos(self):
        from grpc.tools import command
        command.build_package_protos(self.distribution.package_dir[''])


class InstallWithProtos(install, PackgeProtoBuilderMixin):
    def run(self):
        self.build_package_protos()
        install.run(self)


class DevelopWithProtos(develop, PackgeProtoBuilderMixin):
    def run(self):
        self.build_package_protos()
        develop.run(self)


setup(
    name='tensorflow_serving_python',
    version='0.1',
    description='Python client for tensorflow serving',
    author="Sebastian Schlecht",
    license="MIT",
    packages=['tensorflow_serving_python', 'tensorflow_serving_python.protos',
              'tensorflow_serving_python.tensor_utils'],
    package_dir={'': 'src'},
    setup_requires=build_requires,
    install_requires=[
        'cython',
        'grpcio',
        'grpcio-tools',
        'numpy'
    ],
    cmdclass={
        'install': InstallWithProtos,
        'develop': DevelopWithProtos,
    }
)
