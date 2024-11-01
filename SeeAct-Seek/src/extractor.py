import os
import json
import re
import time
import requests
import asyncio
import tiktoken
import toml
from copy import deepcopy
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

from playwright.async_api import async_playwright
from dataclasses import dataclass
from demo_utils.browser_helper import (normal_launch_async, normal_new_context_async)
from demo_utils.inference_engine import encode_image
from aioconsole import ainput, aprint


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
    page.reload()

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


@dataclass
class SessionControl:
    pages = []
    cdp_sessions = []
    active_page = None
    active_cdp_session = None
    context = None
    browser = None

session_control = SessionControl()

EXTRACTOR_MODEL = "gpt-4o"



async def extract_vllm(session_control, task, search_motivation, output_prefix="screenshot", step=0.85):
    toml_config_file = "config/demo_mode_seeact_seeker.toml"
    output_dir = "../online_results/compare/outputs/"
    config = toml.load(toml_config_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # first we get the screenshots of the entire webpage
    # async with async_playwright() as playwright:
    #     session_control.browser = await normal_launch_async(playwright)
    #     session_control.context = await normal_new_context_async(session_control.browser,
    #                                             tracing=config["playwright"]["tracing"],
    #                                             storage_state=None,
    #                                             video_path=None,
    #                                             viewport=config["playwright"]["viewport"],
    #                                             trace_screenshots=config["playwright"]["trace"]["screenshots"],
    #                                             trace_snapshots=config["playwright"]["trace"]["snapshots"],
    #                                             trace_sources=config["playwright"]["trace"]["sources"],
    #                                             geolocation=config["playwright"]["geolocation"],
    #                                             locale=config["playwright"]["locale"])
        
    #     session_control.context.on("page", page_on_open_handler)
    #     session_control.active_page = await session_control.context.new_page()
    #     try:
    #         await session_control.active_page.goto(url, wait_until="load")
    #     except Exception as e:
    #         print("Failed to fully load the webpage before timeout")
    #         print(e)
    #     await asyncio.sleep(3)

        # Get the height of the entire page
    page_height = await session_control.active_page.evaluate('document.body.scrollHeight')
    print(f"Page height: {page_height}")

    screenshot_count = 0
    current_position = 0

    while current_position < page_height and screenshot_count < 10:
        # Capture screenshot
        await session_control.active_page.screenshot(path=os.path.join(output_dir, f'{output_prefix}_{screenshot_count}.png'))
        print(f"Captured screenshot: {output_prefix}_{screenshot_count}.png")

        # Scroll down by half the viewport height
        current_position += config["playwright"]["viewport"]["height"] * step
        await session_control.active_page.evaluate(f'window.scrollTo(0, {current_position})')
        
        screenshot_count += 1
        
        # Give some time for the page to load content dynamically if needed
        # await session_control.active_page.wait_for_timeout(1000)  # 1 second

    await session_control.active_page.evaluate("window.scrollTo(0, 0);")
    # await session_control.browser.close()

    prompt = """INSTRUCTION: Based on the website's screenshots provided, extract relevant information for the following task: "{task}".

    motivation for aggregating information from this page: "{search_motivation}"

    Tasks could be multi-hop and information is to be collected over multiple iterations. And the aggregated informtation from this step will be used for aggregating more detailed information in future steps.
    Hence even if the information in the screenshots dont directly answer the query but can help find the answer in future (or has partial information), extract them. 
    Even if the search motivation has information present, you should extract them from the screenshots.

You should only respond in JSON format as described below and ensure the response can be parsed by Python json.loads.
Response Format: 
{{ 
    "thoughts": "details on what the screenshots contain and reason behind the paragraphs aggregated or discarded",
    "paragraphs": [list of paragraphs extracted from the screenshots relevant to the task. Each paragraph should be detailed (and in string format). For each entity (name) denote in bracket who they are in context of the task at hand and the motivation for aggregating information (this helps further information aggregation). If there is no relevant information, you can just return an empty list. Don't put your own knowledge into it.],
}}"""

    inp_prompt = prompt.format(task=task, search_motivation=search_motivation)
    
    user_content = [{
        "type": "text",
        "text": inp_prompt
    }]
    for i in range(0, screenshot_count):
        image_path = os.path.join(output_dir, f'{output_prefix}_{i}.png')
        user_content.append({
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
          }
        })   

    messages = [
                    {"role": "system", "content": "You are an assistant to help extract relevant information from a website given a user task. You are helping a navigation agent to aggregate information for a task. The website's content will be provided to you in the form of a list of screenshots taken sequentially from top to the bottom."}, 
                    {"role": "user", "content": deepcopy(user_content)}
                ]
    extract_start = time.time()
    llm = ChatOpenAI(model=EXTRACTOR_MODEL, max_tokens=2000, temperature=0)
    structured_llm = llm.with_structured_output(method="json_mode", include_raw=True)
    with get_openai_callback() as cb:
        response = structured_llm.invoke(messages)        
        extractor_cost = float(cb.total_cost)
    extract_time = time.time() - extract_start
    print("extractor response")
    print(response["parsed"])
    return response["parsed"]["paragraphs"], extractor_cost
    # if response["parsing_error"]:
    #     return [], False, (extract_time, extractor_cost)
    # else:
    #     data = response["parsed"]
    #     print(data)
    #     if type(data).__name__ == "dict":
    #         if len(data) == 1:
    #             value = list(data.values())[0]
    #             if type(value).__name__ == "list":
    #                 return value, True, (extract_time, extractor_cost)
    #             else:
    #                 return [value], True, (extract_time, extractor_cost)
    #         else:
    #             return [data]
    #     elif type(data).__name__ == "list":
    #         return data, True, (extract_time, extractor_cost)
    #     else:
    #         return [data], True, (extract_time, extractor_cost)
    
# if __name__ == "__main__":
#     url = "https://en.wikipedia.org/wiki/List_of_largest_companies_by_revenue"
#     task = "In what year did the CEOs of each of the top 5 global companies by revenue begin their term as CEO? "


#     vlm_extracted = asyncio.run(extract_vllm(url, task))
#     print(vlm_extracted)
#     # print(vlm_success)
#     # print(vlm_stats)

