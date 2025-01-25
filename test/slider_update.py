import gradio as gr


# Function that returns updated text and slider value
def update_text_and_slider(input_text, slider_value):
    output_text = f"The text is '{input_text}', and the slider value is {slider_value}"
    new_slider_value = (
        slider_value + 10
    )  # Example logic to update the slider value (you can modify it)
    return output_text, new_slider_value


# Create the slider object
slider_object = gr.Slider(minimum=0, maximum=100, step=1, label="Slider")

# Create the Gradio interface
iface = gr.Interface(
    fn=update_text_and_slider,
    inputs=[
        gr.Textbox(label="Enter Text"),
        slider_object,
    ],  # Use the same slider for input
    outputs=[
        gr.Textbox(label="Output Text"),
        slider_object,
    ],  # New slider for output
    live=False,  # Disable live updates; only update when the button is pressed
)

iface.launch()


# https://discuss.huggingface.co/t/is-it-possible-to-change-the-interface-with-a-button/45021
