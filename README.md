# llm-embed-onnx

[![PyPI](https://img.shields.io/pypi/v/llm-embed-onnx.svg)](https://pypi.org/project/llm-embed-onnx/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-embed-onnx?include_prereleases&label=changelog)](https://github.com/simonw/llm-embed-onnx/releases)
[![Tests](https://github.com/simonw/llm-embed-onnx/actions/workflows/test.yml/badge.svg)](https://github.com/simonw/llm-embed-onnx/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-embed-onnx/blob/main/LICENSE)

Run embedding models using ONNX

This LLM plugin is a wrapper around [onnx_embedding_models](https://github.com/taylorai/onnx_embedding_models) by Benjamin Anderson.

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-embed-onnx
```
## Usage

Usage instructions go here.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-embed-onnx
python3 -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
llm install -e '.[test]'
```
To run the tests:
```bash
pytest
```
