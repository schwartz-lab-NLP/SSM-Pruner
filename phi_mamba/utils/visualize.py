import plotly.express as px
import gradio as gr
import torch
from functorch.dim import Tensor
from transformers import AutoTokenizer
try:
    from phi_mamba.modules.lm_head import LMHeadModel
except ImportError:
    LMHeadModel = None
from phi_mamba.modules.modeling_phi_adjusted import PhiForCausalLM

def load_hf_model(model_path):
    model = PhiForCausalLM.from_pretrained(model_path, attn_implementation='eager')
    model.eval()
    return model

def load_mohawk_model(model_path):
    model = LMHeadModel.from_pretrained(model_path)
    model.eval()
    return model

def get_attention_heads(model, input_ids):
    outputs = model(input_ids, output_attentions=True)
    attention_heads = outputs.attentions if hasattr(outputs, "attentions") else outputs["all_attn_matrices"]
    return attention_heads





def visualize_attention(attention_heads):
    figures = []
    for layer_idx, layer_attention in enumerate(attention_heads):
        avg_attention = layer_attention.mean(dim=1)[0]
        fig = px.imshow(
            avg_attention.detach().cpu().to(torch.float16).numpy(),
            labels=dict(x="Sequence Position", y="Sequence Position"),
            title=f"Layer {layer_idx + 1} Average Attention"
        )
        figures.append(fig)
    return figures


def visualize_models(model_path1, model_path2, input_text):
    model1 = load_hf_model(model_path1)
    if LMHeadModel is None:
        model2 = load_hf_model(model_path2)
    else:
        model2 = load_mohawk_model(model_path2)

    tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-1_5')
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    attention_heads1 = get_attention_heads(model1, input_ids)
    attention_heads2 = get_attention_heads(model2, input_ids)

    figures1 = visualize_attention(attention_heads1)
    figures2 = visualize_attention(attention_heads2)
    res = []
    for fig1, fig2 in zip(figures1, figures2):
        res.append(fig1)
        res.append(fig2)
    return res

def visualize_stage_1(attns1, attns2):
    """
    Visualize attention softmax(KQ) matrices of two models side by side
    attns1: attention matrix of model 1 (batch_size, num_heads1, seq_len, seq_len)
    attns2: attention matrix of model 2 (batch_size, num_heads2, seq_len, seq_len)
    """
    figures1 = visualize_attention(attns1)
    figures2 = visualize_attention(attns2)
    res = []
    for fig1, fig2 in zip(figures1, figures2):
        res.append(fig1)
        res.append(fig2)
    return res

def run_stage1_app(attns1, attns2):
    iface = gr.Interface(
        fn=lambda: visualize_stage_1(attns1, attns2),
        inputs=[],
        outputs=[gr.Plot(label=f"Model {i % 2 + 1} AH") for i in range(len(attns1)+len(attns2))],
        description="Visualize Attention Heads of Two Models"
    )

    iface.launch(share=True)


def run_hf_app():
    iface = gr.Interface(
        fn=visualize_models,
        inputs=[
            gr.Textbox(label="Model 1 Path"),
            gr.Textbox(label="Model 2 Path"),
            gr.Textbox(label="Input Text")
        ],
        outputs=[gr.Plot(label=f"Model {i % 2 + 1} AH") for i in range(48)],
        description="Visualize Attention Heads of Two Models"
    )

    iface.launch(share=True)


import plotly.express as px


import torch
import numpy as np
import gradio as gr

import torch
import numpy as np
import plotly.express as px
import gradio as gr
from typing import Union, List


def plot_binary_mask(
        input_tensor: Union[torch.Tensor, np.ndarray],
        colormap: str = "Greys",
        title: str = "Binary Mask Visualization"
) -> List:
    """
    Plot a binary mask using Plotly with enhanced visualization options.

    Args:
        input_tensor: Input tensor or array to visualize
        colormap: Plotly colormap to use for visualization
        title: Title for the plot

    Returns:
        List containing the Plotly figure
    """
    try:
        # Convert to numpy if tensor
        if isinstance(input_tensor, torch.Tensor):
            img_array = input_tensor.detach().cpu().numpy()
        else:
            img_array = input_tensor

        # Ensure values are between 0 and 1
        if img_array.max() > 1.0:
            img_array = img_array / 255.0

        # Convert to uint8 for display
        img_array = (img_array * 255).astype(np.uint8)

        # Create the figure with enhanced styling
        fig = px.imshow(
            img_array,
            labels=dict(x="Sequence Position", y="Sequence Position"),
            title=title,
            color_continuous_scale=colormap,
        )

        # Update layout for better visualization
        fig.update_layout(
            title_x=0.5,
            title_font_size=16,
            width=600,
            height=600,
            coloraxis_showscale=True
        )

        return [fig]

    except Exception as e:
        raise gr.Error(f"Error processing input: {str(e)}")


def run_viz_anything(
        tensor_to_plot: Union[torch.Tensor, np.ndarray],
        port: int = None,
        share: bool = True
) -> None:
    """
    Launch a Gradio interface for visualizing binary masks.

    Args:
        tensor_to_plot: Tensor or array to visualize
        port: Optional port number for the Gradio interface
        share: Whether to create a public URL
    """
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                colormap_input = gr.Dropdown(
                    choices=["Greys", "Viridis", "Plasma", "Inferno"],
                    value="Greys",
                    label="Colormap"
                )
                title_input = gr.Textbox(
                    value="Binary Mask Visualization",
                    label="Plot Title"
                )

        with gr.Row():
            plot_output = gr.Plot(label="Visualization Output")

        def update_plot(colormap, title):
            return plot_binary_mask(tensor_to_plot, colormap, title)

        # Update plot when inputs change
        colormap_input.change(
            update_plot,
            inputs=[colormap_input, title_input],
            outputs=[plot_output]
        )
        title_input.change(
            update_plot,
            inputs=[colormap_input, title_input],
            outputs=[plot_output]
        )

        # Initial plot
        plot_output.value = update_plot("Greys", "Binary Mask Visualization")

    # Launch the interface
    iface.launch(share=share, server_port=port)