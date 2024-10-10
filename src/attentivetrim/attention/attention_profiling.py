import pickle
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import base64
from io import BytesIO


def clean_token(token):
    """ Clean token by removing unwanted characters and trimming spaces. """
    cleaned_token = token.replace('Ġ',  '&nbsp;')  # Assuming 'Ġ' is a placeholder to be removed
    return cleaned_token


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


def colorize(tokens, attention_scores, original_scores):
    """ Colorize tokens based on attention scores and display as HTML with tooltips. """
    print(f"min: {np.min(attention_scores)}, max: {np.max(attention_scores)}")
    normalized_scores = (attention_scores - np.min(attention_scores)) / (
                np.max(attention_scores) - np.min(attention_scores))
    cmap = matplotlib.colormaps['viridis']

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
    for token, score in zip(tokens, normalized_scores):
        cleaned_token = clean_token(token)
        color = matplotlib.colors.rgb2hex(cmap(score)[:3])
        original_score = original_scores[tokens.index(token)]  # Get the original attention score
        # Tooltip HTML
        token_html = f'<div class="tooltip" style="background-color: {color}; color: black;">{cleaned_token}<span class="tooltiptext">Attention Score: {original_score:.5f}</span></div>'
        colored_string += token_html

    # Generate color bar
    color_bar_html = generate_colorbar(cmap, attention_scores)
    return color_bar_html + '<br>' + colored_string


def print_tokens_with_attention_head(attention, tokens, layer, head):
    print("Token idx, token string, Log Attention Score")
    attention_scores = torch.sum(attention[layer][0][head][-10:], dim=0).detach().numpy()
    log_attention_scores = np.log(attention_scores + 1e-7)  # Adding a small constant to avoid log(0)

    for i, (token, score) in enumerate(zip(tokens, log_attention_scores)):
        cleaned_token = clean_token(token)
        print(f"{i}: {cleaned_token}, {score}, {attention_scores[i]}")

    html_string = colorize(tokens, log_attention_scores, attention_scores)
    with open(f'../data/html/attention_colorized.html', 'w') as f:
        f.write(html_string)

def print_tokens_with_attention_layer(attention, tokens, layer):
    """ Print tokens along with log-transformed attention scores for a specific layer and head, and generate a color-coded HTML output. """
    print("Token idx, token string, Log Attention Score")
    att = attention[layer]
    # add up all the heads
    sum_att = torch.sum(att, dim=(0,1))
    print(f"Summed attention shape: {sum_att[-10:]}")
    attention_scores = torch.sum(sum_att[-10:], dim=0).detach().numpy()
    print(f"Attention scores shape: {attention_scores.shape}")
    log_attention_scores = np.log(attention_scores + 1e-9)  # Adding a small constant to avoid log(0)

    for i, (token, score) in enumerate(zip(tokens, log_attention_scores)):
        cleaned_token = clean_token(token)  # Clean each token before printing
        print(f"{i}: {cleaned_token}, {score}")

    # Generate colorized HTML string
    html_string = colorize(tokens, log_attention_scores, attention_scores)
    with open(f'../data/html/attention_colorized_layer.html', 'w') as f:
        f.write(html_string)


def print_tokens_with_attention_whole(attention, tokens):
    """ Print tokens along with log-transformed attention scores aggregated across all layers, and generate a color-coded HTML output. """
    print("Token idx, token string, Log Attention Score")
    total_attention = torch.zeros_like(attention[0][0][0])  # Initialize with the shape of one head's attention

    # Sum attention across all layers and heads
    for layer in attention:
        sum_att = torch.sum(layer, dim=(0, 1))  # Sum over all heads
        total_attention += sum_att

    # Focus on the last ten tokens
    attention_scores = torch.sum(total_attention[-10:], dim=0).detach().numpy()
    print(f"Attention scores shape: {attention_scores.shape}")
    log_attention_scores = np.log(attention_scores + 1e-9)  # Adding a small constant to avoid log(0)

    for i, (token, score) in enumerate(zip(tokens, log_attention_scores)):
        cleaned_token = clean_token(token)
        print(f"{i}: {cleaned_token}, {score}")


    html_string = colorize(tokens, log_attention_scores, attention_scores)
    with open(f'../data/html/attention_colorized_whole.html', 'w') as f:
        f.write(html_string)

def print_tokens_with_attention_whole_diff(attention_a, tokens_a, attention_b, tokens_b):

    total_attention_a = torch.zeros_like(attention_a[0][0][0])  # Initialize with the shape of one head's attention

    # Sum attention across all layers and heads
    for layer in attention_a:
        sum_att = torch.sum(layer, dim=(0, 1))  # Sum over all heads
        total_attention_a += sum_att

    # Focus on the last ten tokens
    attention_scores_a = torch.sum(total_attention_a[-10:], dim=0).detach().numpy()
    print(f"Attention scores a shape: {attention_scores_a.shape}")

    total_attention_b = torch.zeros_like(attention_b[0][0][0])  # Initialize with the shape of one head's attention

    # Sum attention across all layers and heads
    for layer in attention_b:
        sum_att = torch.sum(layer, dim=(0, 1))  # Sum over all heads
        total_attention_b += sum_att

    # Focus on the last ten tokens
    attention_scores_b = torch.sum(total_attention_b[-10:], dim=0).detach().numpy()
    print(f"Attention scores b shape: {attention_scores_b.shape}")
    # check attention_scores shape and truncate the larger one to the smaller one and then subtract them to get the difference
    if attention_scores_a.shape[0] > attention_scores_b.shape[0]:
        attention_scores_a = attention_scores_a[:attention_scores_b.shape[0]]
        tokens_a = tokens_a[:attention_scores_b.shape[0]]
    else:
        attention_scores_b = attention_scores_b[:attention_scores_a.shape[0]]
        tokens_b = tokens_b[:attention_scores_a.shape[0]]

    attention_diff = attention_scores_a - attention_scores_b

    for i, (token, score) in enumerate(zip(tokens_a, attention_diff)):
        cleaned_token = clean_token(token)
        print(f"{i}: {cleaned_token}, {score}")


    html_string = colorize(tokens_a[1:-12], attention_diff[1:-12], attention_diff[1:-12])
    with open(f'../data/html/attention_colorized_whole_diff.html', 'w') as f:
        f.write(html_string)


def load_data(context_index, question_index):
    """ Load attention weights and tokens from files. """
    with open(f'../data/tensor/context{context_index}-question{question_index}_attention.pkl', 'rb') as f:
        attention = pickle.load(f)
    with open(f'../data/tensor/context{context_index}-question{question_index}_tokens.pkl', 'rb') as f:
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
    grds = [[{"grd_s": 3, "grd_e": 16}, {"grd_s": 3, "grd_e": 18}, {"grd_s": 3, "grd_e": 18}, {"grd_s": 3, "grd_e": 12}, {"grd_s": 3, "grd_e": 11}],
            [{"grd_s": 18, "grd_e": 66}, {"grd_s": 24, "grd_e": 74}, {"grd_s": 20, "grd_e": 65}, {"grd_s": 14, "grd_e": 36}, {"grd_s": 13, "grd_e": 31} ],]
    context_index, question_index = 1, 0
    attention, tokens = load_data(context_index, question_index)

    print(f"Attention shape: {len(attention)}")
    print(f"Tokens length: {len(tokens)}")

    top_10_layers = analyze_attention_layers(attention, grd_s=grds[question_index][context_index]["grd_s"], grd_e=grds[question_index][context_index]["grd_e"])
    print("Top 10 attention layers:")
    for item in top_10_layers:
        print(item)

    top_10_heads = analyze_attention_heads(attention, grd_s=grds[question_index][context_index]["grd_s"], grd_e=grds[question_index][context_index]["grd_e"])
    print("Top 10 attention heads:")
    for item in top_10_heads:
        print(item)

    print_tokens_with_attention_head(attention, tokens, 10, 31)
    print_tokens_with_attention_layer(attention, tokens, 9)
    print_tokens_with_attention_whole(attention, tokens)

    attention_b, tokens_b = load_data(1, 1)
    attention_a, tokens_a = load_data(1, 0)
    print_tokens_with_attention_whole_diff(attention_a, tokens_a, attention_b, tokens_b)


if __name__ == "__main__":
    main()