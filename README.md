# INFOGENT: An Agent-Based Framework for Web Information Aggregation

INFOGENT is a modular framework designed to enable robust web information aggregation. Unlike traditional web agents focused on single-goal navigation, INFOGENT’s unique architecture facilitates complex multi-source data gathering by leveraging autonomous components specialized in search, extraction, and aggregation.

## Overview

INFOGENT redefines web navigation for information-seeking tasks. It enables an agent to gather and aggregate information from various sources to answer complex queries. The framework supports both direct API-based access and interactive visual access modes, making it adaptable to a range of information retrieval scenarios.

### Key Features

- **Navigator**: Conducts searches and identifies relevant websites.
- **Extractor**: Extracts relevant information from the selected web pages.
- **Aggregator**: Aggregates the gathered data and ensures comprehensive coverage.

INFOGENT is modular and feedback-driven, with each component working in tandem to ensure high-quality information gathering and aggregation across diverse web environments.

## Installation

To set up INFOGENT in the interactive visual access setting, clone the repository and install the required dependencies:

```bash
git clone <>
cd infogent/SeeAct-Seek
pip install -r requirements.txt
```
Run infogent in the interactive visual access setting with the following command
```bash
python seeact_seeker.py -c config/demo_mode_seeact_seeker.toml
```
