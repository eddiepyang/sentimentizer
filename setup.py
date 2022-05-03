from setuptools import setup, find_packages

setup(
    name="torch-sentiment",
    version="0.41",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "seaborn",
        "spacy",
        "scikit-learn",
        "jsonlines",
        "gensim",
        "structlog"
    ],
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.rst"],
    },
    entry_points={"console_scripts": ["run=yelp_nlp.rnn.driver:main"]},
    # metadata to display on PyPI
    author="Edward Yang",
    author_email="edwardpyang@gmail.com",
    description="utils to help with project",
    project_urls={
        "Source Code": "https://github.com/eddiepyang/yelp-nlp",
    },
    classifiers=["License :: OSI Approved :: Python Software Foundation License"],
)
