"""Main application for the MarqetSim interface."""

import logging

import gradio as gr
from tinytroupe.agent import TinyPerson
from tinytroupe.examples import (
    create_lisa_the_data_scientist,
    create_marcos_the_physician,
    create_oscar_the_architect,
)


def get_simulation(situation, agent_name, options, adjustable, adjustment_config):

    situation = situation.strip()
    agent_name = agent_name.strip()
    options = options.strip()

    if agent_name == "Lisa":
        agent_sim = (
            TinyPerson.all_agents["Lisa"]
            if TinyPerson.has_agent("Lisa")
            else create_lisa_the_data_scientist()
        )
    elif agent_name == "Oscar":
        agent_sim = (
            TinyPerson.all_agents["Oscar"]
            if TinyPerson.has_agent("Oscar")
            else create_oscar_the_architect()
        )
    elif agent_name == "Marcos":
        agent_sim = (
            TinyPerson.all_agents["Marcos"]
            if TinyPerson.has_agent("Marcos")
            else create_marcos_the_physician()
        )

    else:
        agent_sim = (
            TinyPerson.all_agents["Lisa"]
            if TinyPerson.has_agent("Lisa")
            else create_lisa_the_data_scientist()
        )

    # for opt in adjustment:
    if adjustable:
        for opt in adjustment_config:
            agent_sim.define(opt, adjustment_config[opt])

    agent_sim.change_context(situation)
    result = agent_sim.listen_and_act(options, return_actions=True)

    print(f"\n\n result: {result[-2]['action']['content']}")

    return result[-2]["action"]["content"].strip()


def show_config_toggle(agent, n_configs):
    if agent:
        return [gr.update(visible=True)] * n_configs
    return [gr.update(visible=False)] * n_configs


def set_agent_config_boxes():
    age_opt = gr.Slider(
        minimum=15,
        maximum=60,
        label="Age",
        interactive=True,
        visible=False,
        value=17,
        step=1,
    )

    nationality_opt = gr.Dropdown(
        ["Indonesian", "Malaysian", "Singaporean", "Canadian", "Brazilian"],
        label="Nationality",
        visible=False,
    )
    return [age_opt, nationality_opt]


def set_config_as_json(age, nationality):
    config_json = {"age": age, "nationality": nationality}
    return config_json


with gr.Blocks(
    title="MarqetSim - An LLM Agent Simulation based for Online Advertising.",
    analytics_enabled=True,
) as demo:
    gr.Markdown("# MarqetSim: A simulation based for Online Advertising.")
    gr.Markdown("Demo - Online Advertisement Evaluation for Online Travel Agents")

    with gr.Row(equal_height=True):
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

            # Agent Config
            # nationality, occupation, gender
            agent_config_bool = gr.Checkbox(label="Configure the agent")
            config_boxes = set_agent_config_boxes()
            agent_adjustment = gr.JSON(visible=False)

            agent_config_bool.change(
                show_config_toggle,
                inputs=[
                    agent_config_bool,
                    gr.Number(len(config_boxes), visible=False),
                ],
                outputs=config_boxes,
            )

            agent_config_bool.change(
                set_config_as_json,
                inputs=config_boxes,
                outputs=agent_adjustment,
            )

            greet_btn = gr.Button("Simulate")

        with gr.Column():
            output = gr.Textbox(
                value="The agent opinion about the ads will be shown here.",
                label="Results",
                lines=17,
            )

    gr.Examples(
        examples=[
            [
                "Lisa",
                "Long holiday is near, you want to book your vacation, you see ads on OTA.",
                """Can you evaluate these vacation ads for me? Which one convices you more
                to go for vacation. Select **ONLY** one. Please explain your reasoning, 
                based on your financial situation, background and personality.
                #Ad 1 - Bali, Sunset Beach, Surfing, and Night Club 
                #Ad 2 - Yogyakarta, traditional arts and cultural heritage 
                #Ad 3 - Bandung, large city set amid volcanoes and tea plantations""",
            ],
            [
                "Oscar",
                "Long holiday is near, you want to book your vacation, you see ads on OTA.",
                """Can you evaluate these vacation ads for me? Which one convices you more
                to go for vacation. Select **ONLY** one. Please explain your reasoning, 
                based on your financial situation, background and personality.
                #Ad 1 - Bali, Sunset Beach, Surfing, and Night Club 
                #Ad 2 - Yogyakarta, traditional arts and cultural heritage 
                #Ad 3 - Bandung, large city set amid volcanoes and tea plantations""",
            ],
            [
                "Marcos",
                "Long holiday is near, you want to book your vacation, you see ads on OTA.",
                """Can you evaluate these vacation ads for me? Which one convices you more
                to go for vacation. Select **ONLY** one. Please explain your reasoning, 
                based on your financial situation, background and personality.
                #Ad 1 - Bali, Sunset Beach, Surfing, and Night Club 
                #Ad 2 - Yogyakarta, traditional arts and cultural heritage 
                #Ad 3 - Bandung, large city set amid volcanoes and tea plantations""",
            ],
        ],
        inputs=[agent, situation, options],
        # outputs=[output],
        # fn=get_simulation,
        cache_examples=False,
        label="Try examples",
    )

    greet_btn.click(
        fn=get_simulation,
        inputs=[situation, agent, options, agent_config_bool, agent_adjustment],
        outputs=output,
        api_name="get_simulation",
    )


if __name__ == "__main__":
    demo.launch()
