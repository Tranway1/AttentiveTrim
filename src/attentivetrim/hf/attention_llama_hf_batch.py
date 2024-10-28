# -*- coding: utf-8 -*-
import base64
import json
import gc
import math
import os
from io import BytesIO
from math import factorial

import matplotlib
import numpy as np
from gitdb.fun import chunk_size
from matplotlib import pyplot as plt
from matplotlib.colorbar import ColorbarBase
from torch.onnx.symbolic_opset11 import chunk
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch

from pathlib import Path
HOME_DIR = Path.home()


def print_gpu_memory():
    allocated_memory = torch.cuda.memory_allocated() / 1024**2
    cached_memory = torch.cuda.memory_reserved() / 1024**2
    print("Allocated memory: {:.2f} MB".format(allocated_memory))
    print("Cached memory: {:.2f} MB".format(cached_memory))


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal. Must be a NumPy array.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less than `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    rate: int
        the rate of the derivative to compute (default = 1)

    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or its n-th derivative).

    Raises
    ------
    ValueError
        If `y` is not a NumPy array or if `window_size` and `order` are not integers.
    TypeError
        If `window_size` is not a positive odd number or too small for the polynomial order.
    """
    if not isinstance(y, np.ndarray):
        raise ValueError("Input signal y must be a NumPy array.")

    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomial order")

    order_range = range(order + 1)
    half_window = (window_size - 1) // 2

    # Precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)

    # Pad the signal at the extremes with values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))

    # Convolve the padded signal with the coefficients
    ys = np.convolve(m[::-1], y, mode='valid')
    return ys


def smooth_attention(attention, window=7, order=3):
    smoothed_attention = savitzky_golay(attention, window, order)
    return smoothed_attention


def clean_token(token):
    """ Clean token by removing unwanted characters and trimming spaces. """
    cleaned_token = token.replace('Ġ',  '&nbsp;')  # Assuming 'Ġ' is a placeholder to be removed
    return cleaned_token

def get_token_range_from_char_range(tokens, start_char, end_char):
    assert start_char < end_char, "start_char should be less than end_char"
    print(f"original start_char: {start_char}, original end_char: {end_char}")

    total_chars = sum(len(token) for token in tokens)
    if start_char < 0:
        start_char = total_chars + start_char
    if end_char <= 0:
        end_char = total_chars + end_char

    # Ensure the adjusted indices are within bounds
    start_char = max(0, min(start_char, total_chars))
    end_char = max(0, min(end_char, total_chars))

    print(f"adjusted start_char: {start_char}, adjusted end_char: {end_char}")

    start_token = end_token = None
    char_count = 0
    for idx, token in enumerate(tokens):
        token_length = len(token)
        if start_token is None and char_count + token_length > start_char:
            print(f"start_token: {start_token}, char_count: {char_count}, token_length: {token_length}, index: {idx}， token: {token}")
            start_token = idx
        if end_token is None and char_count + token_length >= end_char:
            print(f"end_token: {end_token}, char_count: {char_count}, token_length: {token_length}, index: {idx}， token: {token}")
            end_token = idx+1
            break
        char_count += token_length

    return start_token, end_token

def generate_colorbar(cmap, attention_scores):
    """ Generate a color bar image and return it as a base64 encoded string for HTML embedding. """
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    norm = matplotlib.colors.Normalize(vmin=np.min(attention_scores), vmax=np.max(attention_scores))
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_label('Attention Score in Log Scale')

    # Convert plot to PNG image
    png_output = BytesIO()
    plt.savefig(png_output, format='png', bbox_inches='tight')
    plt.close(fig)
    png_output.seek(0)  # rewind to beginning of file

    # Encode PNG image to base64 string
    base64_png = base64.b64encode(png_output.getvalue()).decode('utf-8')
    return f'<img src="data:image/png;base64,{base64_png}"/>'


def generate_dual_colorbar(cmap_neg, cmap_pos, min_val, max_val, title="Attention Score"):
    """
    Generate a color bar with negative values in one color map and positive in another.

    Args:
    cmap_neg (matplotlib.colors.Colormap): The colormap for negative values.
    cmap_pos (matplotlib.colors.Colormap): The colormap for positive values.
    min_val (float): Minimum value for the color bar (should be negative).
    max_val (float): Maximum value for the color bar (should be positive).
    title (str): Title for the color bar.

    Returns:
    str: HTML representation of the color bar.
    """
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    norm = matplotlib.colors.TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)
    cb1 = ColorbarBase(ax, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        'Custom', [cmap_neg(1.0), (1, 1, 1, 1), cmap_pos(1.0)]),
        norm=norm,
        orientation='horizontal')
    cb1.set_label(title)
    plt.close(fig)

    # Convert Matplotlib figure to HTML
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    import base64
    data = base64.b64encode(buf.read()).decode('utf-8')
    html = f'<img src="data:image/png;base64,{data}" />'

    return html

def colorize(tokens, attention_scores, original_scores, titlestr="Attention Scores", smooth=True):
    """ Colorize tokens based on attention scores and display as HTML with tooltips. """
    print(f"min: {np.min(attention_scores)}, max: {np.max(attention_scores)}")
    normalized_scores = (attention_scores - np.min(attention_scores)) / (
                np.max(attention_scores) - np.min(attention_scores))
    cmap = matplotlib.colormaps['cividis']
    if smooth:
        normalized_scores = smooth_attention(normalized_scores)

    # HTML and CSS for tooltip
    tooltip_css = """
    <style>
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 120px;
            background-color: black;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px 0;
            position: absolute;
            z-index: 1;
            bottom: 150%;
            left: 50%;
            margin-left: -60px;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
        }
    </style>
    """

    colored_string = tooltip_css  # Start with the CSS for tooltips
    cnt = 0
    for token, score in zip(tokens, normalized_scores):
        cleaned_token = clean_token(token)
        color = matplotlib.colors.rgb2hex(cmap(score)[:3])
        original_score = original_scores[tokens.index(token)]  # Get the original attention score
        # Tooltip HTML
        token_html = f'<div class="tooltip" style="background-color: {color}; color: black;">{cleaned_token}<span class="tooltiptext">Attention index: {cnt}<br> Score: {original_score:.5f}</span></div>'
        colored_string += token_html
        cnt += 1

    # Generate color bar
    color_bar_html = generate_colorbar(cmap, attention_scores)
    # encapsult the titlestring in a div and add on the top of color_bar_html
    header_html = f'<div style="text-align: center; font-size: 20px; font-weight: bold;">{titlestr}</div>' + color_bar_html

    return header_html + '<br>' + colored_string


def colorize_distance_from_zero(tokens, attention_scores, original_scores, titlestr="Attention Scores", smooth=True):
    print(f"min: {np.min(attention_scores)}, max: {np.max(attention_scores)}")

    # Separate positive and negative scores
    positive_scores = np.maximum(0, attention_scores)
    negative_scores = np.minimum(0, attention_scores)

    # Normalize scores
    max_pos = np.max(positive_scores)
    max_neg = np.abs(np.min(negative_scores))

    normalized_pos_scores = positive_scores / max_pos if max_pos != 0 else positive_scores
    normalized_neg_scores = negative_scores / max_neg if max_neg != 0 else negative_scores

    # Choose colormaps
    pos_cmap = matplotlib.cm.get_cmap('Reds')
    neg_cmap = matplotlib.cm.get_cmap('Blues')

    if smooth:
        normalized_pos_scores = smooth_attention(normalized_pos_scores)
        normalized_neg_scores = smooth_attention(normalized_neg_scores)

    # HTML and CSS for tooltip
    tooltip_css = """
    <style>
    .tooltip {
        position: relative;
        display: inline-block;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: black;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px 0;
        position: absolute;
        z-index: 1;
        bottom: 150%;
        left: 50%;
        margin-left: -60px;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
    }
    </style>
    """

    colored_string = tooltip_css
    cnt = 0
    for token, pos_score, neg_score, original_score in zip(tokens, normalized_pos_scores, normalized_neg_scores,
                                                           original_scores):
        if original_score >= 0:
            color = matplotlib.colors.rgb2hex(pos_cmap(pos_score)[:3])
        else:
            color = matplotlib.colors.rgb2hex(neg_cmap(np.abs(neg_score))[:3])

        cleaned_token = clean_token(token)

        # Tooltip HTML
        token_html = f'<div class="tooltip" style="background-color: {color}; color: black;">{cleaned_token}<span class="tooltiptext">Attention index: {cnt}<br> Score: {original_score:.5f}</span></div>'
        colored_string += token_html
        cnt += 1

    # Generate dual color bar
    min_val = np.min(attention_scores)
    max_val = np.max(attention_scores)
    color_bar_html = generate_dual_colorbar(neg_cmap, pos_cmap, min_val, max_val)

    header_html = f'<div style="text-align: center; font-size: 20px; font-weight: bold;">{titlestr}</div>' + color_bar_html

    return header_html + '<br>' + colored_string


def print_tokens_with_attention_head(dataset, attention, tokens, layer, head, question_token_length=10,q_idx=0, c_idx=0, chunk_idx=None):
    print("Token idx, token string, Log Attention Score")
    attention_scores = torch.sum(attention[layer][0][head][-question_token_length:], dim=0).detach().numpy()
    # normalize the attention scores - divide by question_token_length
    attention_scores = attention_scores / question_token_length
    log_attention_scores = np.log(attention_scores + 1e-7)  # Adding a small constant to avoid log(0)
    # output the token indices with Top 50 attention scores
    attention_scores_tensor = torch.from_numpy(attention_scores)
    # topindices = torch.topk(attention_scores_tensor, 50).indices
    # print(f"Top 50 attention scores head: {topindices}")

    html_string = colorize(tokens[1:], log_attention_scores[1:], attention_scores[1:], titlestr=f"Attention Scores for Layer {layer}, Head {head}")
    if chunk_idx is None:
        with open(f'../data/html/full/{dataset}_{c_idx}_{q_idx}_attention_colorized_head_{layer}_{head}.html', 'w') as f:
            f.write(html_string)
    else:
        with open(f'../data/html/{dataset}_{c_idx}_{q_idx}_attention_colorized_head_{layer}_{head}_{chunk_idx}.html', 'w') as f:
            f.write(html_string)
    return attention_scores

def print_tokens_with_attention_layer(dataset, attention, tokens, layer, question_token_length=10,q_idx=0, c_idx=0, chunk_idx=None):
    """ Print tokens along with log-transformed attention scores for a specific layer and head, and generate a color-coded HTML output. """
    print("Token idx, token string, Log Attention Score")
    att = attention[layer]
    # add up all the heads
    sum_att = torch.sum(att, dim=(0,1))
    print(f"Summed attention shape: {sum_att[-question_token_length:]}")
    attention_scores = torch.sum(sum_att[-question_token_length:], dim=0).detach().numpy()
    # normalize the attention scores - divide by question_token_length
    attention_scores = attention_scores / question_token_length
    print(f"Attention scores shape: {attention_scores.shape}")
    log_attention_scores = np.log(attention_scores + 1e-9)  # Adding a small constant to avoid log(0)

    attention_scores_tensor = torch.from_numpy(attention_scores)
    # topindices = torch.topk(attention_scores_tensor, 50).indices
    # print(f"Top 50 attention scores layer: {topindices}")

    # Generate colorized HTML string
    html_string = colorize(tokens[1:], log_attention_scores[1:], attention_scores[1:], titlestr=f"Attention Scores for Layer {layer}")
    if chunk_idx is None:
        with open(f'../data/html/full/{dataset}_{c_idx}_{q_idx}_attention_colorized_layer_{layer}.html', 'w') as f:
            f.write(html_string)
    else:
        with open(f'../data/html/{dataset}_{c_idx}_{q_idx}_attention_colorized_layer_{layer}_{chunk_idx}.html', 'w') as f:
            f.write(html_string)
    return attention_scores


def print_tokens_with_attention_whole(dataset, attention, tokens, question_token_length=10,q_idx=0, c_idx=0, chunk_idx=None):
    """ Print tokens along with log-transformed attention scores aggregated across all layers, and generate a color-coded HTML output. """
    print("Token idx, token string, Log Attention Score")
    total_attention = torch.zeros_like(attention[0][0][0])  # Initialize with the shape of one head's attention

    # Sum attention across all layers and heads
    for layer in attention:
        sum_att = torch.sum(layer, dim=(0, 1))  # Sum over all heads
        total_attention += sum_att

    # Focus on the last ten tokens
    attention_scores = torch.sum(total_attention[-question_token_length:], dim=0).detach().numpy()
    # normalize the attention scores - divide by question_token_length
    attention_scores = attention_scores / question_token_length
    print(f"Attention scores shape: {attention_scores.shape}")
    log_attention_scores = np.log(attention_scores + 1e-9)  # Adding a small constant to avoid log(0)

    attention_scores_tensor = torch.from_numpy(attention_scores)
    # topindices = torch.topk(attention_scores_tensor, 50).indices
    # print(f"Top 50 attention scores for whole: {topindices}")


    html_string = colorize(tokens[1:], log_attention_scores[1:], attention_scores[1:], titlestr="Attention Scores for Whole Layers")
    if chunk_idx is None:
        with open(f'../data/html/full/{dataset}_{c_idx}_{q_idx}_attention_colorized_whole.html', 'w') as f:
            f.write(html_string)
    else:
        with open(f'../data/html/{dataset}_{c_idx}_{q_idx}_attention_colorized_whole_{chunk_idx}.html', 'w') as f:
            f.write(html_string)
    return attention_scores

def print_tokens_with_scores(dataset, attention_scores, tokens, mode, c_idx, q_idx, layer=0, head=0):
    """ Print tokens along with attention scores and generate a color-coded HTML output. """
    print("Token idx, token string, Attention Score")
    print(f"Attention scores shape: {attention_scores.shape}")

    # Adding a small constant to avoid log(0)
    log_attention_scores = np.log(attention_scores + 1e-9)
    file_path =''
    # Generate colorized HTML string
    if mode == "head":
        html_string = colorize(tokens[1:], log_attention_scores[1:], attention_scores[1:], titlestr=f"Attention Scores for Layer {layer}, Head {head}")
        file_path = f'../data/html/full/{dataset}_{c_idx}_{q_idx}_attention_colorized_head_{layer}_{head}.html'
        with open(file_path, 'w') as f:
            f.write(html_string)
    elif mode == "layer":
        html_string = colorize(tokens[1:], log_attention_scores[1:], attention_scores[1:], titlestr=f"Attention Scores for Layer {layer}")
        file_path = f'../data/html/full/{dataset}_{c_idx}_{q_idx}_attention_colorized_layer_{layer}.html'
        with open(file_path, 'w') as f:
            f.write(html_string)
    elif mode == "whole":
        html_string = colorize(tokens[1:], log_attention_scores[1:], attention_scores[1:], titlestr="Attention Scores for Whole Layers")
        file_path = f'../data/html/full/{dataset}_{c_idx}_{q_idx}_attention_colorized_whole.html'
        with open(file_path, 'w') as f:
            f.write(html_string)
    elif mode == "selected":
        html_string = colorize(tokens[1:], log_attention_scores[1:], attention_scores[1:], titlestr="Attention Scores for Selected Heads")
        file_path = f'../data/html/full/{dataset}_{c_idx}_{q_idx}_attention_colorized_selected_heads.html'
        with open(file_path, 'w') as f:
            f.write(html_string)
    print(f"HTML file with mode {mode} saved at: {file_path}")

def print_tokens_with_selected_attention_heads(dataset, attention, tokens, selected_heads, question_token_length=10,q_idx=0, c_idx=0, chunk_idx=None):
    """
    Print tokens along with log-transformed attention scores aggregated from selected attention heads,
    and generate a color-coded HTML output.

    Args:
    attention (Tensor): The attention tensor with dimensions [num_layers, num_heads, seq_length, seq_length].
    tokens (List[str]): List of token strings corresponding to the input sequence.
    selected_heads (List[Tuple[int, int]]): List of tuples where each tuple contains (layer_index, head_index).
    question_token_length (int): Number of tokens from the end to focus on.
    """
    print("Token idx, token string, Log Attention Score")

    # Initialize with the shape of one head's attention, filled with zeros
    selected_attention = torch.zeros_like(attention[0][0][0])

    # Sum attention across specified layers and heads
    for layer_idx, head_idx in selected_heads:
        selected_attention += attention[layer_idx][0][head_idx]

    # Focus on the last ten tokens
    attention_scores = torch.sum(selected_attention[-question_token_length:], dim=0).detach().numpy()

    # Normalize the attention scores - divide by question_token_length
    attention_scores = attention_scores / question_token_length
    print(f"Attention scores shape: {attention_scores.shape}")

    log_attention_scores = np.log(attention_scores + 1e-9)  # Adding a small constant to avoid log(0)
    attention_scores_tensor = torch.from_numpy(attention_scores)
    # topindices = torch.topk(attention_scores_tensor, 50).indices
    # print(f"Top 50 attention scores for selected heads: {topindices}")

    # Generate colorized HTML string
    html_string = colorize(tokens[1:], log_attention_scores[1:], attention_scores[1:],
                           titlestr="Attention Scores for Selected Heads")
    if chunk_idx is None:
        with open(f'../data/html/full/{dataset}_{c_idx}_{q_idx}_attention_colorized_selected_heads.html', 'w') as f:
            f.write(html_string)
    else:
        with open(f'../data/html/{dataset}_{c_idx}_{q_idx}_attention_colorized_selected_heads_{chunk_idx}.html', 'w') as f:
            f.write(html_string)
    return attention_scores

def print_tokens_with_trimmed_attention_whole_diff(dataset, attention_a, tokens_a, attention_b, tokens_b, qa_idx=0, qb_idx=0, c_idx=0, context_ends=10, notes="", chunk_idx=None):

    attention_scores_a = attention_a

    print(f"Attention scores a shape: {attention_scores_a.shape}")

    attention_scores_b = attention_b
    print(f"Attention scores b shape: {attention_scores_b.shape}")
    # check attention_scores shape and truncate the larger one to the smaller one and then subtract them to get the difference
    context_boundary = context_ends
    print(f"Context boundary: {context_boundary}")

    attention_scores_a = attention_scores_a[:context_boundary]
    attention_scores_b = attention_scores_b[:context_boundary]
    attention_diff = attention_scores_a - attention_scores_b

    # for i, (token, score) in enumerate(zip(tokens_a, attention_diff)):
    #     cleaned_token = clean_token(token)
    #     print(f"{i}: {cleaned_token}, {score}")
    attention_scores_tensor = torch.from_numpy(attention_diff)
    # topindices = torch.topk(attention_scores_tensor, 50, largest=False).indices
    # print(f"Top 50 attention scores by differential attention: {topindices}")

    html_string = colorize_distance_from_zero(tokens_a[1:context_boundary], attention_diff[1:context_boundary], attention_diff[1:context_boundary], titlestr=f"Attention Scores Difference between questions Att{qa_idx}-Att{qb_idx}<br>{notes}")
    # html_string = colorize(tokens_a[1:context_boundary], attention_diff[1:context_boundary], attention_diff[1:context_boundary], titlestr=f"Attention Scores Difference between questions Att{qa_idx}-Att{qb_idx}<br>{notes}")
    if chunk_idx is None:
        with open(f'../data/html/full/{dataset}_{c_idx}_{qa_idx}_{qb_idx}_attention_colorized_whole_diff.html', 'w') as f:
            f.write(html_string)
    else:
        with open(f'../data/html/{dataset}_{c_idx}_{qa_idx}_{qb_idx}_attention_colorized_whole_diff_{chunk_idx}.html', 'w') as f:
            f.write(html_string)
    return attention_diff




def process_context_with_chunks(context, question, tokenizer, model, token_dir, attention_summary_dir, dataset, file_idx, query_idx, top_10_heads=None, top_10_layers=None, chunk_size=10000, overlap=50, whole_only=False):
    chunked_context = []
    q_offset = 0
    print(f"Processing file {file_idx} with chunking instead of the whole context")
    # Chunk the context
    for i in range(0, len(context), chunk_size - overlap):
        chunked_context.append(context[i:i + chunk_size])
        print(f"chunk {i} starts at {i} and ends at {i + chunk_size}")

    print(f"Total chunks: {len(chunked_context)}")
    best_heads = []
    best_layers = []
    selected_heads = []
    wholes = []
    token_arr = []



    # Process each chunk
    for chunk_idx, chunk in enumerate(chunked_context):
        print(f"Processing chunk {chunk_idx} with {len(chunk)} characters")
        prompt = f"Context: {chunk}\nQuestion: {question}\nAnswer:"
        try:
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            outputs = model(inputs)

        except RuntimeError as e:
            print(f"RuntimeError while processing file {file_idx} on chunk {chunk_idx}: {e}")
            print_gpu_memory()
            gc.collect()  # Explicitly invoke garbage collector
            torch.cuda.empty_cache()
            print("Memory freed")
            print_gpu_memory()
            continue

        print(len(outputs))
        attention = outputs[-1]
        tokens = tokenizer.convert_ids_to_tokens(inputs[0])

        print(f"Attention layers: {len(attention)}")
        print(f"Attention map shape: {attention[0][0][0].shape}")
        print(f"tokens length: {len(tokens)}")

        # Get question range in the tokens
        question_section = f"Question: {question}\nAnswer:"
        print(f"{question_section}")
        print(f"Question section length: {len(question_section)}")
        startq_token_idx, endq_token_idx = get_token_range_from_char_range(tokens, -len(question_section), 0)
        print(f"Question token range: {startq_token_idx}, {endq_token_idx}")
        q_offset = endq_token_idx - startq_token_idx


        if not whole_only:

            layer_idx, head_idx = top_10_heads[0]
            best_layer = top_10_layers[0]

            # Example function calls (you need to define these functions or adjust according to your actual use case)
            attention_head = print_tokens_with_attention_head(dataset, attention, tokens, layer_idx, head_idx,
                                                              question_token_length=endq_token_idx - startq_token_idx,
                                                              q_idx=query_idx, c_idx=file_idx, chunk_idx=chunk_idx)
            attention_layer = print_tokens_with_attention_layer(dataset, attention, tokens, best_layer,
                                                                question_token_length=endq_token_idx - startq_token_idx,
                                                                q_idx=query_idx, c_idx=file_idx, chunk_idx=chunk_idx)

            attention_selected = print_tokens_with_selected_attention_heads(dataset, attention, tokens, top_10_heads[:5],
                                                                        question_token_length=endq_token_idx - startq_token_idx,
                                                                        q_idx=query_idx, c_idx=file_idx,
                                                                        chunk_idx=chunk_idx)
            best_heads.append(attention_head)
            best_layers.append(attention_layer)
            selected_heads.append(attention_selected)
        attention_summary = print_tokens_with_attention_whole(dataset, attention, tokens, question_token_length=endq_token_idx - startq_token_idx, q_idx=query_idx, c_idx=file_idx, chunk_idx=chunk_idx)
        wholes.append(attention_summary)
        token_arr.append(tokens)

        # Save tokens and attention summary
        token_file = f"{token_dir}/{dataset}_{file_idx}_{chunk_idx}.json"
        attention_summary_file = f"{attention_summary_dir}/{dataset}_{file_idx}_{query_idx}_{chunk_idx}.npy"

        if query_idx == 0:
            with open(token_file, 'w') as f:
                json.dump(tokens, f)
        with open(attention_summary_file, 'wb') as f:
            np.save(f, attention_summary)

        print_gpu_memory()
        del inputs
        del outputs
        del attention
        del tokens
        gc.collect()  # Explicitly invoke garbage collector
        torch.cuda.empty_cache()
        print("Memory freed")
        print_gpu_memory()

    # get the longest token array and normalize the attention scores by * len(token_arr[i])/max_len
    max_len = max([len(t) for t in token_arr])
    print(f"Normalized by max token length: {max_len}")
    for i in range(len(wholes)):
        print(f"Normalizing chunk {i} with length {len(token_arr[i])} / {max_len}")
        wholes[i] = wholes[i] * len(token_arr[i]) / max_len
        if not whole_only:
            best_heads[i] = best_heads[i] * len(token_arr[i]) / max_len
            best_layers[i] = best_layers[i] * len(token_arr[i]) / max_len
            selected_heads[i] = selected_heads[i] * len(token_arr[i]) / max_len



    # merge all the chunks
    if not whole_only:
        merged_best_heads = best_heads[0][:-q_offset]
        merged_best_layers = best_layers[0][:-q_offset]
        merged_seleted_heads = selected_heads[0][:-q_offset]
    merged_wholes = wholes[0][:-q_offset]
    merged_tokens = token_arr[0][:-q_offset]

    for i in range(1, len(wholes)-1):
        print(f"Merging chunk {i}")
        if not whole_only:

            merged_best_heads = np.concatenate((merged_best_heads, best_heads[i][3:-q_offset]))
            merged_best_layers = np.concatenate((merged_best_layers, best_layers[i][3:-q_offset]))
            merged_seleted_heads = np.concatenate((merged_seleted_heads, selected_heads[i][3:-q_offset]))
        merged_wholes = np.concatenate((merged_wholes, wholes[i][3:-q_offset]))
        merged_tokens += token_arr[i][3:-q_offset]

    print(f"processing the last chunk {len(wholes)-1}")

    if not whole_only:
        merged_best_heads = np.concatenate((merged_best_heads, best_heads[-1][3:]))
        merged_best_layers = np.concatenate((merged_best_layers, best_layers[-1][3:]))
        merged_seleted_heads = np.concatenate((merged_seleted_heads, selected_heads[-1][3:]))
    merged_wholes = np.concatenate((merged_wholes, wholes[-1][3:]))
    merged_tokens += token_arr[-1][3:]

    # Draw the html for the merged tokens
    if not whole_only:
        print_tokens_with_scores(dataset, merged_best_heads, merged_tokens, "head", file_idx, query_idx, top_10_heads[0][0], top_10_heads[0][1])
        print_tokens_with_scores(dataset, merged_best_layers, merged_tokens, "layer", file_idx, query_idx, top_10_layers[0])
        print_tokens_with_scores(dataset, merged_seleted_heads, merged_tokens, "selected", file_idx, query_idx)
    print_tokens_with_scores(dataset, merged_wholes, merged_tokens, "whole", file_idx, query_idx)

    return merged_wholes, merged_tokens



def process_file_question(dataset, query_idx, question, file_idx, context, tokenizer, model, token_dir, attention_summary_dir,
                          top_10_heads_parse=None, top_10_layers_parse=None, chunk_size=10000, overlap=50, whole_only=False):
    global outputs, inputs
    print(f"Processing file {file_idx} with question: {question}")
    if context.strip():
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer:"
    inputs = None
    outputs = None
    tokens = None
    attention = None

    enable_chunking = False
    if len(context) < chunk_size:
        try:
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            outputs = model(inputs)
            print(f"Processing the full context {file_idx} as a whole")
        except RuntimeError as e:
            print_gpu_memory()
            enable_chunking = True
            print(f"RuntimeError while processing the full context  {file_idx} as a whole: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            print("Memory freed")
            print_gpu_memory()
    else:
        enable_chunking = True

    if enable_chunking:
        print_gpu_memory()
        # if the context is too long, we need to chunk it
        merged_whole, merged_tokens = process_context_with_chunks(context, question, tokenizer, model, token_dir,
                                                                  attention_summary_dir,
                                                                  dataset, file_idx, query_idx,
                                                                  top_10_heads=top_10_heads_parse,
                                                                  top_10_layers=top_10_layers_parse,
                                                                  chunk_size=chunk_size, overlap=overlap, whole_only=whole_only)
        tokens = merged_tokens
        attention_summary = merged_whole
    else:
        print(len(outputs))
        attention = outputs[-1]
        tokens = tokenizer.convert_ids_to_tokens(inputs[0])
        # save the attention weights into a file

        print(f"Attention shape: {len(attention)}")
        print(f"tokens length: {len(tokens)}")

        # get question range in the tokens
        question_section = f"Question: {question}\nAnswer:"
        print(f"{question_section}")
        print(f"Question section length: {len(question_section)}")
        startq_token_idx, endq_token_idx = get_token_range_from_char_range(tokens, -len(question_section), 0)
        print(f"Question token range: {startq_token_idx}, {endq_token_idx}")
        if not whole_only:
            assert top_10_heads_parse is not None
            layer_idx, head_idx = top_10_heads_parse[0]
            best_layer = top_10_layers_parse[0]
            print_tokens_with_attention_head(dataset, attention, tokens, layer_idx, head_idx,
                                             question_token_length=endq_token_idx - startq_token_idx, q_idx=query_idx,
                                             c_idx=file_idx)
            print_tokens_with_attention_layer(dataset, attention, tokens, best_layer,
                                              question_token_length=endq_token_idx - startq_token_idx, q_idx=query_idx,
                                              c_idx=file_idx)
            print_tokens_with_selected_attention_heads(dataset, attention, tokens, top_10_heads_parse[:5],
                                                       question_token_length=endq_token_idx - startq_token_idx,
                                                       q_idx=query_idx, c_idx=file_idx)

        attention_summary = print_tokens_with_attention_whole(dataset, attention, tokens,
                                                              question_token_length=endq_token_idx - startq_token_idx,
                                                              q_idx=query_idx, c_idx=file_idx)

    token_file = f"{token_dir}/{dataset}_{file_idx}.json"
    attention_summary_file = f"{attention_summary_dir}/{dataset}_{file_idx}_{query_idx}.npy"

    if query_idx == 0:
        with open(token_file, 'w') as f:
            json.dump(tokens, f)
    with open(attention_summary_file, 'wb') as f:
        np.save(f, attention_summary)
    print_gpu_memory()
    # Explicitly delete attention and tokens to free up memory
    if inputs is not None:
        del inputs
    if outputs is not None:
        del outputs
    if attention is not None:
        del attention
    if tokens is not None:
        del tokens
    gc.collect()  # Explicitly invoke garbage collector
    torch.cuda.empty_cache()
    print("Memory freed")
    print_gpu_memory()



def run_dataset_analysis(model_name, dataset, whole_only = False, chunk_size=10000, overlap=50):
    # 0: <|begin_of_text|>, 17
    # 1: Context, 7
    # 2: :, 1
    offset = 25

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True, load_in_4bit=True, device_map="auto")
    print(f"Model loaded: {model_name}")

    config_path = '../../../questions/question.json'
    with open(config_path) as f:
        data = json.load(f)

    file_list = data["datasets"][dataset]["list"]
    questions = data["query"][f'{dataset.upper()}_QUESTIONS']
    head_stats = data["head_stats"]
    layer_stats = data["layer_stats"]
    token_dir = "../data/tokens"
    attention_summary_dir = "../data/attention_summary"
    # create if not exist
    os.makedirs(token_dir, exist_ok=True)
    os.makedirs(attention_summary_dir, exist_ok=True)

    for query_idx, question in enumerate(questions):
        print_gpu_memory()
        layer_count = {}
        head_count = {}
        print(f"Processing question: {question} on dataset: {dataset}")


        top_10_layers = layer_stats[dataset].get(question, {})
        top_10_heads = head_stats[dataset].get(question, {})
        # each entry in the layers like: "Layer 0": 10
        # each entry in the heads like: "Layer: 0, Head: 0": 10
        top_10_layers_parse = []
        top_10_heads_parse = []
        for key, value in top_10_layers.items():
            print(f"key: {key}, value: {value}")
            layer_idx = int(key.split(' ')[1])
            top_10_layers_parse.append(layer_idx)

        for key, value in top_10_heads.items():
            print(f"key: {key}, value: {value}")
            layer_idx = int(key.split(' ')[1].split(',')[0])
            head_idx = int(key.split(' ')[3])
            top_10_heads_parse.append((layer_idx, head_idx))

        if len(top_10_heads_parse) == 0 or len(top_10_layers_parse) == 0:
            whole_only = True

        for file_idx, file_add in enumerate(file_list):
            print(f"Processing file {file_idx}: {file_add}")

            file_path = file_add.replace("/Users/chunwei/research/", f"{HOME_DIR}/")
            if HOME_DIR == "/home/chunwei":
                file_path = (file_add.replace("/Users/chunwei/research/", f"{HOME_DIR}/").replace("(", "\\( ").replace(")", "\\)"))
            with open(file_path, 'r') as f:
                json_obj = json.loads(f.read())
            context = json_obj["symbols"]


            # call the function to process the context per file and question
            process_file_question(dataset, query_idx, question, file_idx, context, tokenizer, model, token_dir, attention_summary_dir,
                                    top_10_heads_parse=top_10_heads_parse, top_10_layers_parse=top_10_layers_parse, chunk_size=chunk_size, overlap=overlap, whole_only=whole_only)

            # Run baseline query processing with the first question
            if query_idx == 0:
                # process the baseline query
                process_file_question(dataset, -1, "Please repeat the context.", file_idx, context, tokenizer, model, token_dir, attention_summary_dir,
                                      top_10_heads_parse=top_10_heads_parse, top_10_layers_parse=top_10_layers_parse, chunk_size=chunk_size, overlap=overlap, whole_only=True)


    print(f"Done with {dataset} set and {len(questions)} questions")




def run_differential_attention(model_name, dataset, chunk_size=10000, overlap=50):
    config_path = '../../../questions/question.json'
    with open(config_path) as f:
        data = json.load(f)

    print(f"Model loaded: {model_name}")

    file_list = data["datasets"][dataset]["list"]
    questions = data["query"][f'{dataset.upper()}_QUESTIONS']
    token_dir = "../data/tokens"
    attention_summary_dir = "../data/attention_summary"
    # create if not exist
    os.makedirs(token_dir, exist_ok=True)
    os.makedirs(attention_summary_dir, exist_ok=True)

    cos_sim_percentages = data["cos_sim"][dataset]
    faraway_query = []
    for cur_idx, cos_sim_percentage in enumerate(cos_sim_percentages):
        # get the index with smallest cosine similarity
        min_idx = np.argmin(cos_sim_percentage)
        faraway_query.append(min_idx)

    for file_idx, f_path in enumerate(file_list):
        print(f"Processing file {file_idx}: {f_path}")

        for query_idx, question in enumerate(questions):
            farest_query_idx = faraway_query[query_idx]
            print(f"Processing question: {question} on dataset: {dataset}")
            token_file = f"{token_dir}/{dataset}_{file_idx}.json"
            cur_attention_summary_file = f"{attention_summary_dir}/{dataset}_{file_idx}_{query_idx}.npy"
            farmost_attention_summary_file = f"{attention_summary_dir}/{dataset}_{file_idx}_{farest_query_idx}.npy"
            repeat_attention_summary_file = f"{attention_summary_dir}/{dataset}_{file_idx}_-1.npy"

            with open(token_file, 'r') as f:
                tokens = json.load(f)

            # Get question range in the tokens
            question_section = f"Question: {questions[0]}\nAnswer:"
            print(f"{question_section}")
            print(f"Question section length: {len(question_section)}")
            startq_token_idx, endq_token_idx = get_token_range_from_char_range(tokens, -len(question_section), 0)
            print(f"Question token range: {startq_token_idx}, {endq_token_idx}")
            # check if attention summary file exists
            if os.path.exists(cur_attention_summary_file) and os.path.exists(farmost_attention_summary_file):
                print(f"comparing with question {query_idx} with {farest_query_idx}")
                with open(cur_attention_summary_file, 'rb') as f:
                    cur_attention_summary = np.load(f)
                with open(farmost_attention_summary_file, 'rb') as f:
                    farest_attention_summary = np.load(f)

                attention_diff = print_tokens_with_trimmed_attention_whole_diff(dataset, cur_attention_summary, tokens,
                                                                               farest_attention_summary, tokens,
                                                                               qa_idx=query_idx, qb_idx=farest_query_idx,
                                                                               c_idx=file_idx, context_ends = startq_token_idx-1,
                                                                                notes=f"Qa +: {question}, Qb -: {questions[farest_query_idx]}")
                # save the attention summary
                res_attention_summary_file = f"{attention_summary_dir}/{dataset}_{file_idx}_{query_idx}_{farest_query_idx}.npy"
                with open(res_attention_summary_file, 'wb') as f:
                    np.save(f, attention_diff)
                # Explicitly delete attention and tokens to free up memory
                del cur_attention_summary
                del farest_attention_summary




            if os.path.exists(repeat_attention_summary_file) and os.path.exists(cur_attention_summary_file):
                print(f"comparing with question {query_idx} with repeat baseline")
                with open(cur_attention_summary_file, 'rb') as f:
                    cur_attention_summary = np.load(f)
                with open(repeat_attention_summary_file, 'rb') as f:
                    repeat_attention_summary = np.load(f)
                attention_diff = print_tokens_with_trimmed_attention_whole_diff(dataset, cur_attention_summary, tokens,
                                                                               repeat_attention_summary, tokens,
                                                                               qa_idx=query_idx, qb_idx=-1,
                                                                               c_idx=file_idx, context_ends = startq_token_idx-1,
                                                                                notes=f"Qa +: {question}, Qb -: Repeat the context")
                # save the attention summary
                res_attention_summary_file = f"{attention_summary_dir}/{dataset}_{file_idx}_{query_idx}_-1.npy"
                with open(res_attention_summary_file, 'wb') as f:
                    np.save(f, attention_diff)
                # Explicitly delete attention and tokens to free up memory
                del cur_attention_summary
                del repeat_attention_summary
            del tokens
            gc.collect()
            torch.cuda.empty_cache()




if __name__ == "__main__":
    # 0: < | begin_of_text | >, 17
    # 1: Context, 7
    # 2::, 1
    # local_model_path = f'{HOME_DIR}/hf/Mistral-7B-Instruct-v0.2'
    # local_model_path = f'{HOME_DIR}/hf/dbrx-instruct/'
    model_name = f"{HOME_DIR}/hf/Llama-3.2-1B-Instruct"

    # dataset = "notice"
    dataset = "paper"
    run_dataset_analysis(model_name, dataset, chunk_size=8000, overlap=50)
    print("================= Done with dataset analysis, now starting differential attention analysis =================")
    run_differential_attention(model_name, dataset,chunk_size=8000, overlap=50)