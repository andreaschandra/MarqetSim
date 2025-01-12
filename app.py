"""Main application for the MarqetSim interface."""

import gradio as gr
from utils.config import Config
from tinytroupe.agent import TinyPerson
from tinytroupe.examples import (
    create_lisa_the_data_scientist,
    create_oscar_the_architect,
    create_marcos_the_physician,
)
import logging


def get_simulation(situation, agent_name, options):

    situation = situation.strip()
    agent_name = agent_name.strip()
    options = options.strip()

    if agent_name == "Lisa the data scientist":
        agent = create_lisa_the_data_scientist()
    elif agent_name == "Oscar the architect":
        agent = create_oscar_the_architect()
    elif agent_name == "Marcos the physician":
        agent = create_marcos_the_physician()

    agent.change_context(situation)
    result = agent.listen_and_act(options, return_actions=True)

    print(f"\n\n result: {result[-2]['action']['content']}")

    return result[-2]["action"]["content"].strip()


with gr.Blocks(
    title="MarqetSim - An LLM Simulation based for product discovery.",
    analytics_enabled=True,
) as demo:
    gr.Markdown("# MarqetSim: A simulation based for product discovery.")
    gr.Markdown("Demo - Online Advertisement Evaluation for TVs")
    gr.Markdown("## Example for Advertisement for TV.")
    gr.Markdown("### Situation")
    gr.Markdown(
        """
        ```
        Your TV broke and you need a new one. You search for a new TV on Bing.
        ```
        """
    )
    gr.Markdown("### Agents")
    gr.Markdown(
        """
        ```
        Lisa the data scientist
        ```
        """
    )
    gr.Markdown("### Options")
    gr.Markdown(
        """
        ```
        Can you evaluate these Bing ads for me? Which one convices you more to buy their particular offering? 
        Select **ONLY** one. Please explain your reasoning, based on your financial situation, background and personality.

        # AD 1

        The Best TV Of Tomorrow - LG 4K Ultra HD TV
        https://www.lg.com/tv/oled
        AdThe Leading Name in Cinematic Picture. Upgrade Your TV to 4K OLED And See The Difference. It's Not Just OLED, It's LG OLED. Exclusive a9 Processor, Bringing Cinematic Picture Home.

        Infinite Contrast · Self-Lighting OLED · Dolby Vision™ IQ · ThinQ AI w/ Magic Remote

        Free Wall Mounting Deal
        LG G2 97" OLED evo TV
        Free TV Stand w/ Purchase
        World's No.1 OLED TV

        # AD 2

        The Full Samsung TV Lineup - Neo QLED, OLED, 4K, 8K & More
        https://www.samsung.com
        AdFrom 4K To 8K, QLED To OLED, Lifestyle TVs & More, Your Perfect TV Is In Our Lineup. Experience Unrivaled Technology & Design In Our Ultra-Premium 8K & 4K TVs.

        Discover Samsung Event · Real Depth Enhancer · Anti-Reflection · 48 mo 0% APR Financing

        The 2023 OLED TV Is Here
        Samsung Neo QLED 4K TVs
        Samsung Financing
        Ranked #1 By The ACSI®

        # AD 3

        Wayfair 55 Inch Tv - Wayfair 55 Inch Tv Décor
        Shop Now
        https://www.wayfair.com/furniture/free-shipping
        AdFree Shipping on Orders Over $35. Shop Furniture, Home Décor, Cookware & More! Free Shipping on All Orders Over $35. Shop 55 Inch Tv, Home Décor, Cookware & More!

        ``` 
        """
    )
    gr.Markdown("## Situation")
    gr.Markdown(
        "Your TV broke and you need a new one. You search for a new TV on Bing."
    )
    gr.Markdown("## Profile")
    gr.Markdown("Lisa the data scientist")

    situation = gr.Textbox(label="Situation", info="Describe agent situation")
    agent = gr.Dropdown(
        ["Lisa the data scientist", "Oscar the architect", "Marcos the physician"],
        label="Agent",
        info="Pick the agent you want to simulate.",
    )
    options = gr.Textbox(
        label="Options", info="Enter the options you want to simulate."
    )

    output = gr.Textbox(label="Results")

    greet_btn = gr.Button("Simulate")
    greet_btn.click(
        fn=get_simulation,
        inputs=[situation, agent, options],
        outputs=output,
        api_name="get_simulation",
    )


if __name__ == "__main__":
    demo.launch()
