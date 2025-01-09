# EPUBKit

EPUBKit is a Python toolkit for analyzing and processing EPUB files, with a focus on extracting structured content based on HTML patterns.

## Features

- Extract content by category (headers, subheaders, body text, etc.)
- Interactive GUI for selecting and training content patterns
- Support for complex nested HTML structures
- Handles calibre-style EPUB formatting
- Pattern-based content classification
- Support for footnotes and special formatting

## Installation

### From Source

To install EPUBKit from source, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/EPUBKit.git
   ```
2. Navigate to the project directory:
   ```sh
   cd EPUBKit
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### From PyPI

To install EPUBKit from PyPI, use the following command:
```sh
pip install epubkit
```

## Usage

To use EPUBKit, you can start by importing the library and loading an EPUB file:
```python
import epubkit

# Load an EPUB file
epub = epubkit.load('path/to/your/file.epub')

# Extract content based on patterns
content = epub.extract_content()
```

For more detailed usage instructions and examples, please refer to the documentation.
