#!/usr/bin/env python
from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


mmcls_version_file = 'mmcls/version.py'
mmdet_version_file = 'mmdet/version.py'
mmseg_version_file = 'mmseg/version.py'


def mmcls_get_version():
    with open(mmcls_version_file, 'r') as f:
        exec(compile(f.read(), mmcls_version_file, 'exec'))
    return locals()['__version__']


def mmdet_get_version():
    with open(mmdet_version_file, 'r') as f:
        exec(compile(f.read(), mmdet_version_file, 'exec'))
    return locals()['__version__']


def mmseg_get_version():
    with open(mmseg_version_file, 'r') as f:
        exec(compile(f.read(), mmseg_version_file, 'exec'))
    return locals()['__version__']


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    import re
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


def mmclsbuild():
    setup(
        name='mmcls',
        version=mmcls_get_version(),
        description='Open MMLab Detection Toolbox and Benchmark',
        long_description=readme(),
        author='OpenMMLab',
        author_email='chenkaidev@gmail.com',
        keywords='computer vision, image classification',
        url='https://github.com/open-mmlab/mmclassification',
        packages=find_packages(
            exclude=('configs', 'configs_custom', 'tools', 'demo', 'mmdet',
                     'mmseg')),
        package_data={'mmcls.ops': ['*/*.so']},
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        license='Apache License 2.0',
        setup_requires=parse_requirements('requirements/build.txt'),
        tests_require=parse_requirements('requirements/tests.txt'),
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
            'build': parse_requirements('requirements/build.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
        },
        ext_modules=[],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)


def mmdetbuild():
    setup(
        name='mmdet',
        version=mmdet_get_version(),
        description='Open MMLab Detection Toolbox and Benchmark',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='OpenMMLab',
        author_email='openmmlab@gmail.com',
        keywords='computer vision, object detection',
        url='https://github.com/open-mmlab/mmdetection',
        packages=find_packages(
            exclude=('configs', 'configs_custom', 'tools', 'demo', 'mmcls',
                     'mmseg')),
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        license='Apache License 2.0',
        setup_requires=parse_requirements('requirements/build.txt'),
        tests_require=parse_requirements('requirements/tests.txt'),
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
            'build': parse_requirements('requirements/build.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
        },
        ext_modules=[],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)


def mmsegbuild():
    setup(
        name='mmseg',
        version=mmseg_get_version(),
        description='Open MMLab Detection Toolbox and Benchmark',
        long_description=readme(),
        author='OpenMMLab',
        author_email='chenkaidev@gmail.com',
        keywords='computer vision, semantic segmentation',
        url='https://github.com/open-mmlab/mmsegmentation',
        packages=find_packages(
            exclude=('configs', 'configs_custom', 'tools', 'demo', 'mmcls',
                     'mmdet')),
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        license='Apache License 2.0',
        setup_requires=parse_requirements('requirements/build.txt'),
        tests_require=parse_requirements('requirements/tests.txt'),
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
            'build': parse_requirements('requirements/build.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
        },
        ext_modules=[],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)


if __name__ == '__main__':
    # mmclsbuild()
    mmdetbuild()
    # mmsegbuild()
