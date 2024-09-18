# RAG Chain Project

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chain using LangChain and Chroma. It allows users to ask questions about specific technologies mentioned in provided web articles, retrieving relevant information based on the context.

## Features

- Loads documents from specified URLs.
- Splits documents into manageable chunks for processing.
- Utilizes embeddings for efficient retrieval of relevant information.
- Interactive command-line interface for user queries.

## Requirements

- Python 3.x
- Required libraries:
  - `chromadb`
  - `langchain_community`
  - `langchain_core`
  - `langchain_ollama`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Nahuelluca20/ollama-vector-db
   cd ollama-vector-db
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Update the `urls` list in `main.py` with the desired web pages.
2. Run the application:

   ```bash
   python main.py
   ```

3. Type your question in the command line interface. Type `/exit` to terminate the program.

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes.

## License

This project is licensed under the MIT License.
