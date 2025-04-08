from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    '''Return a list of requirements from the given file.'''
    with open(file_path, 'r') as file_obj:
        requirements = [req.strip() for req in file_obj.readlines()]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name="MLProject",
    version="0.0.1",
    author="Gayanath Perera",
    author_email="gayanath009@yahoo.com", 
    packages=find_packages(),
    description="A machine learning project",
    install_requires=get_requirements('requirements.txt'),
)
