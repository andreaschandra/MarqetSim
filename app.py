"""Main application for the MarqetSim interface."""

import logging

import gradio as gr
from tinytroupe.agent import TinyPerson
from tinytroupe.examples import (
    create_lisa_the_data_scientist,
    create_marcos_the_physician,
    create_oscar_the_architect,
)


def get_simulation(situation, agent_name, options, adjustment):

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

    for opt in adjustment:
        agent_sim.define(opt, adjustment[opt])

    agent_sim.change_context(situation)
    result = agent_sim.listen_and_act(options, return_actions=True)

    print(f"\n\n result: {result[-2]['action']['content']}")
    print(adjustment)
    print(agent_sim._configuration["age"])

    return result[-2]["action"]["content"].strip()


def get_adjustment_config(age_opt):
    print("Get Adjustment")
    conf = {"age": age_opt}
    return conf


def toggle_slider(agent):
    # Show the slider only if an agent is selected
    if agent:
        return gr.update(visible=True)
    return gr.update(visible=False)


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

            age_opt = gr.Slider(
                minimum=15,
                maximum=60,
                label="Age Slider",
                interactive=True,
                visible=False,
            )
            agent.change(toggle_slider, inputs=agent, outputs=age_opt)

            # nationality, occupation, gender
            greet_btn = gr.Button("Simulate")

        with gr.Column():
            output = gr.Textbox(
                value="The agent opinion about the ads will be shown here.",
                label="Results",
                lines=25,
            )

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
        # outputs=[output],
        # fn=get_simulation,
        cache_examples=False,
        label="Try examples",
    )

    # agent_adjustment = gr.JSON()
    # gr.Interface(
    #     fn=get_adjustment_config,
    #     inputs=age_opt,
    #     outputs=agent_adjustment,  # Output the JSON-like object
    # )
    # print(type(agent_adjustment))

    greet_btn.click(
        fn=get_simulation,
        inputs=[situation, agent, options],
        outputs=output,
        api_name="get_simulation",
    )


if __name__ == "__main__":
    demo.launch()


# Lisa : Bali, Jogja
# Oscar : Jogja, Jogja
# Marcos : Jogja
