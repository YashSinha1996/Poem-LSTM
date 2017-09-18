from generate_words import GenerativeNetwork
from random import randint

net = GenerativeNetwork(
    "../sonnets.txt", "model_words.yaml", "weights_words.hdf5")


# def format_sonnet(text):
#     return text
#     formatted = text.split("\n")

#     # The first and last line cut off in the middle, so we'll ditch them
#     formatted = formatted[1:len(formatted) - 1]

#     # Eliminate empty strings, strings that are just newlines, or other
#     # improper strings
#     formatted = [string for string in formatted if len(string) > 3]

#     # Put a period on our last string, replacing other punctuation if it's
#     # there.
#     if formatted[-1][-1].isalnum():
#         formatted[-1] += "."
#     else:
#         formatted[-1] = formatted[-1][:-1] + "."

#     return formatted


# def tag_seed(seed):
#     # Grab a chunk of three words
#     word_list = seed.split()
#     i = randint(1, len(word_list) - 3)

#     bad_start_end = set(['on', 'of', 'from', "I", "O!",
#                          "and", "be", 'or', 'the', 'than', 'with', 'by'])
#     bad_start = set(['of'])
#     bad_end = set(['no', 'an', 'if'])

#     words = []
#     for i, word in enumerate(word_list[i:i + 3]):
#         if not word == "I" and not word == "O!":
#             word = word.strip("',.;-!:?").lower()
#         if i == 0 and word not in bad_start_end | bad_start:
#             words.append(word)
#         if i == 1:
#             words.append(word)
#         if i == 2 and word not in bad_start_end | bad_end:
#             words.append(word)

#     tag = " ".join(words)
#     return tag


def sonnet(old_seed_phrase="", error=5):
    old_seed_tag = old_seed_phrase
    # old_seed = net.make_seed(old_seed_phrase)

    message = net.generate(old_seed_phrase, error)

    return message

if __name__ == '__main__':
    poem = sonnet(error=0)
    for line in poem:
        print(line)
