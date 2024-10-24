import pickle
import torch
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import base64
from io import BytesIO
import numpy as np
from math import factorial

from src.attentivetrim.hf.attention_llama_hf import questions


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


def colorize(tokens, attention_scores, original_scores, titlestr="Attention Scores", smooth=True):
    """ Colorize tokens based on attention scores and display as HTML with tooltips. """
    print(f"min: {np.min(attention_scores)}, max: {np.max(attention_scores)}")
    normalized_scores = (attention_scores - np.min(attention_scores)) / (
                np.max(attention_scores) - np.min(attention_scores))
    cmap = matplotlib.colormaps['viridis']
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


def print_tokens_with_attention_head(attention, tokens, layer, head, question_token_length=10):
    print("Token idx, token string, Log Attention Score")
    attention_scores = torch.sum(attention[layer][0][head][-question_token_length:], dim=0).detach().numpy()
    # normalize the attention scores - divide by question_token_length
    attention_scores = attention_scores / question_token_length
    log_attention_scores = np.log(attention_scores + 1e-7)  # Adding a small constant to avoid log(0)
    # output the token indices with Top 50 attention scores
    attention_scores_tensor = torch.from_numpy(attention_scores)
    topindices = torch.topk(attention_scores_tensor, 50).indices
    print(f"Top 50 attention scores head: {topindices}")



    html_string = colorize(tokens[1:], log_attention_scores[1:], attention_scores[1:], titlestr=f"Attention Scores for Layer {layer}, Head {head}")
    with open(f'../data/html/attention_colorized.html', 'w') as f:
        f.write(html_string)

def print_tokens_with_attention_layer(attention, tokens, layer, question_token_length=10):
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
    topindices = torch.topk(attention_scores_tensor, 50).indices
    print(f"Top 50 attention scores layer: {topindices}")

    # Generate colorized HTML string
    html_string = colorize(tokens[1:], log_attention_scores[1:], attention_scores[1:], titlestr=f"Attention Scores for Layer {layer}")
    with open(f'../data/html/attention_colorized_layer.html', 'w') as f:
        f.write(html_string)


def print_tokens_with_attention_whole(attention, tokens, question_token_length=10):
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
    topindices = torch.topk(attention_scores_tensor, 50).indices
    print(f"Top 50 attention scores for whole: {topindices}")


    html_string = colorize(tokens[1:], log_attention_scores[1:], attention_scores[1:], titlestr="Attention Scores for Whole Layers")
    with open(f'../data/html/attention_colorized_whole.html', 'w') as f:
        f.write(html_string)


def print_tokens_with_selected_attention_heads(attention, tokens, selected_heads, question_token_length=10):
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
    topindices = torch.topk(attention_scores_tensor, 50).indices
    print(f"Top 50 attention scores for selected heads: {topindices}")

    # Generate colorized HTML string
    html_string = colorize(tokens[1:], log_attention_scores[1:], attention_scores[1:],
                           titlestr="Attention Scores for Selected Heads")

    with open(f'../data/html/attention_colorized_selected_heads.html', 'w') as f:
        f.write(html_string)

def print_tokens_with_attention_whole_diff(attention_a, tokens_a, attention_b, tokens_b, questiona_token_length=10, questionb_token_length=10):

    total_attention_a = torch.zeros_like(attention_a[0][0][0])  # Initialize with the shape of one head's attention

    # Sum attention across all layers and heads
    for layer in attention_a:
        sum_att = torch.sum(layer, dim=(0, 1))  # Sum over all heads
        total_attention_a += sum_att

    # Focus on the last ten tokens
    attention_scores_a = torch.sum(total_attention_a[-questiona_token_length:], dim=0).detach().numpy()
    # normalize the attention scores - divide by questiona_token_length
    attention_scores_a = attention_scores_a / questiona_token_length
    print(f"Attention scores a shape: {attention_scores_a.shape}")

    total_attention_b = torch.zeros_like(attention_b[0][0][0])  # Initialize with the shape of one head's attention

    # Sum attention across all layers and heads
    for layer in attention_b:
        sum_att = torch.sum(layer, dim=(0, 1))  # Sum over all heads
        total_attention_b += sum_att

    # Focus on the last ten tokens
    attention_scores_b = torch.sum(total_attention_b[-questionb_token_length:], dim=0).detach().numpy()
    # normalize the attention scores - divide by questionb_token_length
    attention_scores_b = attention_scores_b / questionb_token_length
    print(f"Attention scores b shape: {attention_scores_b.shape}")
    # check attention_scores shape and truncate the larger one to the smaller one and then subtract them to get the difference
    if attention_scores_a.shape[0] > attention_scores_b.shape[0]:
        attention_scores_a = attention_scores_a[:attention_scores_b.shape[0]]
        tokens_a = tokens_a[:attention_scores_b.shape[0]]
    else:
        attention_scores_b = attention_scores_b[:attention_scores_a.shape[0]]
        tokens_b = tokens_b[:attention_scores_a.shape[0]]

    attention_diff = attention_scores_a - attention_scores_b

    # for i, (token, score) in enumerate(zip(tokens_a, attention_diff)):
    #     cleaned_token = clean_token(token)
    #     print(f"{i}: {cleaned_token}, {score}")
    attention_scores_tensor = torch.from_numpy(attention_diff)
    topindices = torch.topk(attention_scores_tensor, 50, largest=False).indices
    print(f"Top 50 attention scores by differential attention: {topindices}")


    html_string = colorize(tokens_a[1:-12], attention_diff[1:-12], attention_diff[1:-12], titlestr="Attention Scores Difference between question")
    with open(f'../data/html/attention_colorized_whole_diff.html', 'w') as f:
        f.write(html_string)


def load_data(context_index, question_index, model_version=""):
    """ Load attention weights and tokens from files. """
    with open(f'../data/tensor/context{model_version}{context_index}-question{question_index}_attention.pkl', 'rb') as f:
        attention = pickle.load(f)
    with open(f'../data/tensor/context{model_version}{context_index}-question{question_index}_tokens.pkl', 'rb') as f:
        tokens = pickle.load(f)
    return attention, tokens


def analyze_attention_layers(attention, grd_s = 3, grd_e = 16, question_length = 10, k = 30):
    """ Analyze attention across layers focusing on specific token indices. """
    att_layers = []
    for layer in range(len(attention)):
        sum_attention = torch.sum(attention[layer], dim=(0, 1))
        topk = torch.topk(sum_attention[-question_length:], k)
        count = sum(grd_s <= index[i] <= grd_e for index in topk.indices for i in range(k))
        att_layers.append([f"Layer {layer}", count])
    sorted_att_layers = sorted(att_layers, key=lambda x: x[1], reverse=True)
    return sorted_att_layers[:10]


def analyze_attention_heads(attention, grd_s = 3, grd_e = 16, question_length = 10, k = 30):
    """ Analyze attention across heads in each layer focusing on specific token indices. """
    print("Analyzing attention heads...")
    att_heads = []
    print("Grd_s: ", grd_s,     "Grd_e: ", grd_e)
    for layer in range(len(attention)):
        multi_head = torch.sum(attention[layer], dim=0)
        for head in range(len(multi_head)):
            sum_attention = multi_head[head]
            topk = torch.topk(sum_attention[-question_length:], k)
            count = sum(grd_s <= index[i] <= grd_e for index in topk.indices for i in range(k))
            att_heads.append([f"Layer: {layer}, Head: {head}", count])
    sorted_att_heads = sorted(att_heads, key=lambda x: x[1], reverse=True)
    return sorted_att_heads[:10]






def main():
    model_version = "3-2"
    grds = [[{"grd_s": 3, "grd_e": 16}, {"grd_s": 3, "grd_e": 18}, {"grd_s": 3, "grd_e": 18}, {"grd_s": 3, "grd_e": 12}, {"grd_s": 3, "grd_e": 11}],
            [{"grd_s": 18, "grd_e": 66}, {"grd_s": 24, "grd_e": 74}, {"grd_s": 20, "grd_e": 65}, {"grd_s": 14, "grd_e": 36}, {"grd_s": 13, "grd_e": 31} ],]
    context_index, question_index = 8, 1
    question = questions[question_index]
    question_section = f"Question: {question}\nAnswer:"
    print(f"{question_section}")
    print(f"Question section length: {len(question_section)}")
    attention, tokens = load_data(context_index, question_index, model_version=model_version)
    # start_token_idx, end_token_idx = get_token_range_from_char_range(tokens, -len(question_section), 0)
    # print(f"token range for question section: {start_token_idx}, {end_token_idx}")
    # print(f"token in the question section: {tokens[start_token_idx:end_token_idx]}")
    # char_selected = sum([len(token) for token in tokens[start_token_idx:end_token_idx]])
    # print(f"total selected token length: {char_selected}")
    for idx, token in enumerate(tokens):
        print(f"{idx}: {token}, {len(token)}")

    # print(f"Attention shape: {len(attention)}")
    # print(f"Tokens length: {len(tokens)}")
    #
    # top_10_layers = analyze_attention_layers(attention, grd_s=grds[question_index][context_index]["grd_s"], grd_e=grds[question_index][context_index]["grd_e"])
    # print("Top 10 attention layers:")
    # for item in top_10_layers:
    #     print(item)
    #
    # top_10_heads = analyze_attention_heads(attention, grd_s=grds[question_index][context_index]["grd_s"], grd_e=grds[question_index][context_index]["grd_e"])
    # print("Top 10 attention heads:")
    # for item in top_10_heads:
    #     print(item)
    #
    #

    print_tokens_with_attention_head(attention, tokens, 7, 12)
    print_tokens_with_attention_layer(attention, tokens, 8)
    print_tokens_with_attention_whole(attention, tokens)

    attention_b, tokens_b = load_data(8, 1, model_version=model_version)
    attention_a, tokens_a = load_data(8, 0, model_version=model_version)
    print_tokens_with_attention_whole_diff(attention_a, tokens_a, attention_b, tokens_b)


if __name__ == "__main__":
    main()

