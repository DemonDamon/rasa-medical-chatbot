# Date    : 2024/6/2 21:26
# File    : utils.py
# Desc    : 
# Author  : Damon
# E-mail  : bingzhenli@hotmail.com

from datasketch import MinHash

def colorstr(*input, color="blue", bold=False, underline=False):
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    prefix = colors[color]
    if bold:
        prefix += colors["bold"]
    if underline:
        prefix += colors["underline"]

    return "{}{}{}".format(prefix, input[0], colors["end"])

def search_sim_terms(word, standard_words_list, threshold=0.5):
    mh1 = MinHash()
    for ch in word:
        mh1.update(ch.encode('utf8'))

    res = {}
    for sw in standard_words_list:
        mh2 = MinHash()
        for ch in sw:
            mh2.update(ch.encode('utf8'))
        score = mh1.jaccard(mh2)
        if score > threshold:
            res.update({sw: score})
            if score == 1:
                break

    return res
