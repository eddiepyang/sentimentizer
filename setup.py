from setuptools import setup, find_packages

setup(
    name="torch-sentiment",
    version="0.4.2",
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
        "structlog",
        "psutil",
    ],
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.rst"],
    },
    entry_points={"console_scripts": ["run=torch_sentiment.workflows.driver:main"]},
    # metadata to display on PyPI
    author="Edward Yang",
    author_email="edwardpyang@gmail.com",
    description="straight forward rnn model",
    project_urls={
        "Source Code": "https://github.com/eddiepyang/torch-sentiment",
    },
    classifiers=["License :: OSI Approved :: Python Software Foundation License"],
)
