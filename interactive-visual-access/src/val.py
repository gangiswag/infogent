# -*- coding: utf-8 -*-
# Copyright (c) 2024 OSU Natural Language Processing Group
#
# Licensed under the OpenRAIL-S License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.licenses.ai/ai-pubs-open-rails-vz1
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script leverages the GPT-4V API and Playwright to create a web agent capable of autonomously performing tasks on webpages.
It utilizes Playwright to create browser and retrieve interactive elements, then apply [SeeAct Framework](https://osu-nlp-group.github.io/SeeAct/) to generate and ground the next operation.
The script is designed to automate complex web interactions, enhancing accessibility and efficiency in web navigation tasks.
"""

import argparse
import asyncio
import datetime
import json
import logging
import os
import warnings
from dataclasses import dataclass

import toml
import torch
from aioconsole import ainput, aprint
from playwright.async_api import async_playwright

from data_utils.format_prompt_utils import get_index_from_option_name
from data_utils.prompts import generate_prompt, format_options
from demo_utils.browser_helper import (normal_launch_async, normal_new_context_async,
                                       get_interactive_elements_with_playwright, get_text_paragraphs_with_playwright, select_option, saveconfig)
from demo_utils.format_prompt import format_choices, format_ranking_input, postprocess_action_lmm
from demo_utils.inference_engine import OpenaiEngine
from demo_utils.ranking_model import CrossEncoder, find_topk
from demo_utils.website_dict import website_dict

from info_seeker_util import InfoSeeker
from extractor import extract_vllm
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

# Remove Huggingface internal warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class SessionControl:
    pages = []
    cdp_sessions = []
    active_page = None
    active_cdp_session = None
    context = None
    browser = None


session_control = SessionControl()


#
# async def init_cdp_session(page):
#     cdp_session = await page.context.new_cdp_session(page)
#     await cdp_session.send("DOM.enable")
#     await cdp_session.send("Overlay.enable")
#     await cdp_session.send("Accessibility.enable")
#     await cdp_session.send("Page.enable")
#     await cdp_session.send("Emulation.setFocusEmulationEnabled", {"enabled": True})
#     return cdp_session

def get_search_prompt(question, passages):
    prompt = """Based the provided context, answer the following question. Output only your answer. Never refuse to answer. If the answer is a list, output one on each line. For all time sensitive questions, consider current year: 2024.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""

    context = ""
    for index, passage in enumerate(passages):
        context += str(index + 1) + ") " + passage["content"].strip() + "\n"
    
    return prompt.format(context=context, question=question)


def format_final_answer(question, passages):
    llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=4000, temperature=0)
    messages = [
                {"role": "user", "content": get_search_prompt(question, passages)}
            ]
    with get_openai_callback() as cb:
        response = llm.invoke(messages)  

    return response.content
    

async def page_on_close_handler(page):
    # print("Closed: ", page)
    if session_control.context:
        # if True:
        try:
            await session_control.active_page.title()
            # print("Current active page: ", session_control.active_page)
        except:
            await aprint("The active tab was closed. Will switch to the last page (or open a new default google page)")
            # print("All pages:")
            # print('-' * 10)
            # print(session_control.context.pages)
            # print('-' * 10)
            if session_control.context.pages:
                session_control.active_page = session_control.context.pages[-1]
                await session_control.active_page.bring_to_front()
                await aprint("Switched the active tab to: ", session_control.active_page.url)
            else:
                await session_control.context.new_page()
                try:
                    await session_control.active_page.goto("https://www.google.com/", wait_until="load")
                except Exception as e:
                    pass
                await aprint("Switched the active tab to: ", session_control.active_page.url)


async def page_on_navigatio_handler(frame):
    session_control.active_page = frame.page
    # print("Page navigated to:", frame.url)
    # print("The active tab is set to: ", frame.page.url)


async def page_on_crash_handler(page):
    await aprint("Page crashed:", page.url)
    await aprint("Try to reload")
    await page.reload()


async def page_on_open_handler(page):
    # print("Opened: ",page)
    page.on("framenavigated", page_on_navigatio_handler)
    page.on("close", page_on_close_handler)
    page.on("crash", page_on_crash_handler)
    session_control.active_page = page
    # print("The active tab is set to: ", page.url)
    # print("All pages:")
    # print('-'*10)
    # print(session_control.context.pages)
    # print("active page: ",session_control.active_page)
    # print('-' * 10)

async def main(config, base_dir) -> None:
    save_video = config["playwright"]["save_video"]
    viewport_size = config["playwright"]["viewport"]
    tracing = config["playwright"]["tracing"]
    locale = None
    try:
        locale = config["playwright"]["locale"]
    except:
        pass
    geolocation = None
    try:
        geolocation = config["playwright"]["geolocation"]
    except:
        pass
    trace_screenshots = config["playwright"]["trace"]["screenshots"]
    trace_snapshots = config["playwright"]["trace"]["snapshots"]
    trace_sources = config["playwright"]["trace"]["sources"]

    try:
        storage_state = config["basic"]["storage_state"]
    except:
        storage_state = None
        

    
    async with async_playwright() as playwright:
        session_control.browser = await normal_launch_async(playwright)
        session_control.context = await normal_new_context_async(session_control.browser,
                                                                    tracing=tracing,
                                                                    storage_state=storage_state,
                                                                    video_path= None,
                                                                    viewport=viewport_size,
                                                                    trace_screenshots=trace_screenshots,
                                                                    trace_snapshots=trace_snapshots,
                                                                    trace_sources=trace_sources,
                                                                    geolocation=geolocation,
                                                                    locale=locale)
        session_control.context.on("page", page_on_open_handler)
        await session_control.context.new_page()
        await session_control.active_page.goto("https://www.google.com/", timeout=60000, wait_until="load")
        await session_control.active_page.goto("https://www.tripadvisor.com/Restaurants-g312741-zfn7816466-zfz10992-Buenos_Aires_Capital_Federal_District.html", timeout=60000, wait_until="load")
        await session_control.active_page.wait_for_timeout(1000) 
        await session_control.active_page.screenshot(path="./a.jpg", full_page=True,
                                                                     type='jpeg', quality=20, timeout=50000)
        # for i in elements:
        #     print(f"{i}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config_path", help="Path to the TOML configuration file.", type=str, metavar='config',
                        default=f"{os.path.join('config', 'demo_mode_seeact_seeker.toml')}")
    args = parser.parse_args()

    # Load configuration file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = None
    try:
        with open(os.path.join(base_dir, args.config_path) if not os.path.isabs(args.config_path) else args.config_path,
                  'r') as toml_config_file:
            config = toml.load(toml_config_file)
            print(f"Configuration File Loaded - {os.path.join(base_dir, args.config_path)}")
    except FileNotFoundError:
        print(f"Error: File '{args.config_path}' not found.")
    except toml.TomlDecodeError:
        print(f"Error: File '{args.config_path}' is not a valid TOML file.")

    asyncio.run(main(config, base_dir))