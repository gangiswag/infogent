# INFOGENT: An Agent-Based Framework for Web Information Aggregation

INFOGENT is a modular framework designed for web information aggregation. Unlike traditional web agents focused on single-goal navigation, INFOGENT’s unique architecture facilitates complex multi-source data gathering by leveraging autonomous components specialized in navigation, extraction, and aggregation.


## Overview

INFOGENT redefines web navigation for information-seeking tasks. It enables an agent to gather and aggregate information from various sources to answer complex queries. The framework supports both direct API-based access and interactive visual access modes, making it adaptable to a range of information retrieval scenarios. INFOGENT employs a modular, feedback-driven approach to information aggregation, making it suitable for complex queries requiring diverse sources.

<img src="static/images/infogent_teaser.png"  width="55%" height="55%">


- **Navigator**: Conducts web search and identifies relevant websites.
- **Extractor**: Extracts relevant information from the selected web pages.
- **Aggregator**: Aggregates the extracted data and provides feedback to the Navigator.


## Installation

To set up INFOGENT in the interactive visual access setting, clone the repository and install the required dependencies:

```bash
git clone https://github.com/gangiswag/infogent.git
pip install -r requirements.txt
```

## Direct-API Access

The Direct-API Access makes use of OpenAI LLMs and Google Search via the SerperAPI. You can get your search key from [] and setup the environment variable as:

```bash
export OPENAI_API_KEY=<your OpenAI key here>
export SERPER_API_KEY=<your search key here>
```

Before running on FanOutQA, you need to do the following to setup the fanoutqa evaluation library:
```bash
pip install "fanoutqa[eval]"
python -m spacy download en_core_web_sm
pip install "bleurt @ git+https://github.com/google-research/bleurt.git@master"
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
unzip BLEURT-20.zip
rm BLEURT-20.zip
```

To run the information aggregation process on FanOutQA:
```bash
cd direct-api-driven
mkdir -p results
python run_fanoutqa.py \
    --navigator_model gpt-4o-mini \
    --aggregator_model gpt-4o-mini \
    --extractor_model gpt-4o-mini \
    --inp_path data/fanoutqa-final-dev.json \
    --out_path results/fanoutqa_dev_agg.json \
    --log_path results/fanoutqa_dev_agg.log
```

Then, to generate the final answer and run the evaluation:
```bash
python fanoutqa_answer.py \
    --answer_model gpt-4o-mini \
    --data_path data/fanoutqa-final-dev.json \
    --input_path results/fanoutqa_dev_agg.json \
    --out_path results/fanoutqa_dev_answer.json \
    --score_path results/fanoutqa_dev_score.json

```

To evaluate the closed book model, run:
```bash
python fanoutqa_answer.py \
    --answer_model gpt-4o-mini --closed_book \
    --data_path data/fanoutqa-final-dev.json \
    --out_path results/fanoutqa_dev_closed_answer.json \
    --score_path results/fanoutqa_dev_closed_score.json
```



## Interactive Visual Access

To run INFOGENT:
```bash
python seeact_seeker.py -c config/demo_mode.toml
```

## Installation


If you found this repo useful for your work, please consider citing our paper:
```
@article{reddy2024infogent,
        title={Infogent: An Agent-Based Framework for Web Information Aggregation},
        author={Reddy, Revanth Gangi and Mukherjee, Sagnik and Kim, Jeonghwan and Wang, Zhenhailong and Hakkani-Tur, Dilek and Ji, Heng},
        journal={arXiv preprint arXiv:2410.19054},
        year={2024}
      }
```


