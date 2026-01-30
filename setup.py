from setuptools import find_packages,setup
from typing import List
HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    """
    This will return the requiremnets
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines() ## lekin har baar next line pe jaane par \n bhi aaye go so we have tp repalce it 
        requirements = [req.replace("\n","") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements        

setup(
    name="ML-Proj",
    version="0.001",
    author="Adnan",
    author_email="adnanalam6414@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)