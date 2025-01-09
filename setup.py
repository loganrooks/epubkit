from setuptools import setup, find_packages

setup(
    name="epubkit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "beautifulsoup4>=4.12.0",
        "ebooklib>=0.18",
        "openai>=1.3.0",
        "typing_extensions>=4.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "mypy>=1.6.0",
            "black>=23.9.0",
            "isort>=5.12.0",
        ],
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="A toolkit for analyzing and processing EPUB files",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="epub, ebooks, parsing, analysis",
    url="https://github.com/yourusername/epubkit",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    entry_points={
        "console_scripts": [
            "epubkit=epubkit.parser:main",
        ],
    },
)
