from setuptools import setup, find_packages

setup(
    name='face_emotion_lib',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'face_recognition',
        'deepface'
    ],
    description='A library for face recognition and emotion detection',
    author='Osama Abu Shehab',
    author_email='osama99.shehab@outlook.com',
    url=,  # Replace with your repository URL
)

