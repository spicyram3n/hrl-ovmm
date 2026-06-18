from glob import glob

from setuptools import find_packages, setup

package_name = 'fetcher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hsr',
    maintainer_email='poorna.sesetti03@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'good_boy = fetcher.good_boy:main',
            'seeker = fetcher.seeker:main',
            'gazebo_scene_graph = fetcher.gazebo_scene_graph:main',
            'search_and_fetch = fetcher.search_and_fetch_node:main',
        ],
    },
)
