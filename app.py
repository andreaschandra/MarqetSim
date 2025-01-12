"""Main application for the MarqetSim interface."""

import gradio as gr
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

    if agent_name == "Lisa":
        agent = (
            TinyPerson.all_agents["Lisa"]
            if TinyPerson.has_agent("Lisa")
            else create_lisa_the_data_scientist()
        )
    elif agent_name == "Oscar":
        agent = (
            TinyPerson.all_agents["Oscar"]
            if TinyPerson.has_agent("Oscar")
            else create_oscar_the_architect()
        )
    elif agent_name == "Marcos":
        agent = (
            TinyPerson.all_agents["Marcos"]
            if TinyPerson.has_agent("Marcos")
            else create_marcos_the_physician()
        )

    agent.change_context(situation)
    result = agent.listen_and_act(options, return_actions=True)

    print(f"\n\n result: {result[-2]['action']['content']}")

    return result[-2]["action"]["content"].strip()


with gr.Blocks(
    title="MarqetSim - An LLM Agent Simulation based for Online Advertising.",
    analytics_enabled=True,
) as demo:
    gr.Markdown("# MarqetSim: A simulation based for product discovery.")
    gr.Markdown("Demo - Online Advertisement Evaluation for Online Travel Agents")

    with gr.Row():
        with gr.Column():
            agent = gr.Dropdown(
                [
                    "Lisa",
                    "Oscar",
                    "Marcos",
                ],
                label="Agent",
                info="Pick the agent you want to simulate.",
            )
            situation = gr.Textbox(label="Situation", info="Describe agent situation")
            options = gr.Textbox(
                label="Options", info="Enter the options you want to simulate."
            )
            greet_btn = gr.Button("Simulate")

        with gr.Column():
            output = gr.Textbox(label="Results")

    gr.Examples(
        examples=[
            [
                "Lisa",
                "Long holiday is near, you want to book your vacation, you see ads on OTA.",
                "Can you evaluate these vacation ads for me? Which one convices you more to go for vacation. Select **ONLY** one. Please explain your reasoning, based on your financial situation, background and personality.\n#Ad 1 - Bali, Sunset Beach, Surfing, and Night Club \n#Ad 2 - Yogyakarta, traditional arts and cultural heritage \n#Ad 3 - Bandung, large city set amid volcanoes and tea plantations",
            ],
            [
                "Oscar",
                "Long holiday is near, you want to book your vacation, you see ads on OTA.",
                "Can you evaluate these vacation ads for me? Which one convices you more to go for vacation. Select **ONLY** one. Please explain your reasoning, based on your financial situation, background and personality.\n#Ad 1 - Bali, Sunset Beach, Surfing, and Night Club \n#Ad 2 - Yogyakarta, traditional arts and cultural heritage \n#Ad 3 - Bandung, large city set amid volcanoes and tea plantations",
            ],
            [
                "Marcos",
                "Long holiday is near, you want to book your vacation, you see ads on OTA.",
                "Can you evaluate these vacation ads for me? Which one convices you more to go for vacation. Select **ONLY** one. Please explain your reasoning, based on your financial situation, background and personality.\n#Ad 1 - Bali, Sunset Beach, Surfing, and Night Club \n#Ad 2 - Yogyakarta, traditional arts and cultural heritage \n#Ad 3 - Bandung, large city set amid volcanoes and tea plantations",
            ],
        ],
        inputs=[agent, situation, options],
        outputs=[output],
        fn=get_simulation,
        cache_examples=False,
        label="Try examples",
    )

    greet_btn.click(
        fn=get_simulation,
        inputs=[situation, agent, options],
        outputs=output,
        api_name="get_simulation",
    )


if __name__ == "__main__":
    demo.launch()
