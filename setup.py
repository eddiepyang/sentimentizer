from setuptools import setup, find_packages

setup(
    name="yelp-nlp",
    version="0.31",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18",
        "pandas",
        "torch",
        "seaborn",
        "spacy",
        "scikit-learn",
        "jsonlines",
        "pytest",
        "gensim",
    ],
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.rst"],
    },
    entry_points={"console_scripts": ["yelp-nlp=yelp_nlp.rnn.driver:main"]},
    # metadata to display on PyPI
    author="Edward Yang",
    author_email="edwardpyang@gmail.com",
    description="utils to help with project",
    project_urls={
        "Source Code": "https://github.com/eddiepyang/yelp-nlp",
    },
    classifiers=["License :: OSI Approved :: Python Software Foundation License"],
)
