

from setuptools import setup, find_packages
setup(
    name="yelp_nlp",
    version="0.1",
    packages=find_packages(),

    install_requires=[
        "numpy", "pandas", "torch", 
        "seaborn", "spacy", "scikit-learn", 'jsonlines'
    ],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.rst"],
    },

    # metadata to display on PyPI
    author="Edward Yang",
    author_email="eddiepyang@gmail.com",
    description="utils to help with project",
    project_urls={
        "Source Code": "https://github.com/eddiepyang/yelp_nlp",
    },
    classifiers=[
        "License :: OSI Approved :: Python Software Foundation License"
    ]

)

