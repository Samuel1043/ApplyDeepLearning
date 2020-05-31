import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def showAttention(input_sentence, output_words, attentions,show_range=30):
    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(111)
    attentions=attentions[:show_range,:show_range]
    print(len(attentions))
    input_sentence=input_sentence[:show_range+1]
    output_words=output_words[:show_range+1]
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels(input_sentence, rotation=90)
    ax.set_yticklabels(output_words)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
