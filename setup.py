from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
    """
    Function to get the list of requirements from requirements.txt file.
    Returns:
        List[str]: A list of package names required for the project.
    """ 
    requirement_lst:List[str] = []
    try:

        with open('requirements.txt','r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found. Please ensure it exists in the project directory.")
    return requirement_lst

setup(
    name='NetworkSecurity',
    version='0.1',
    author='Kashish',
    author_email="kashishkhera20@gmail.com",
    description='A project for network security analysis',
    packages=find_packages(),
    install_requires=get_requirements()
)