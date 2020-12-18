import preprocessor as p
import re
from string import punctuation

p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY)

hashtag_p = re.compile(
    r"(#stayhomesavelives)|(#KeepLockdown)|(#lockdownnow)|(#StopReopenAmerica)|"
    r"(#DontReopenAmerica)|(#DoNotEndLockdown)|(#ProtestLockdown)|(#ReOpenAmerica)|"
    r"(#Reopen\(statebrv\))|(#LetUsWork)|(#Open\(state\)Now)|(#IamEssential)|"
    r"(#LiberateAmerica)|(#LiberateTheUSA)|(#backtowork)",
    re.IGNORECASE,
)


def _clean_tweet(string):
    string = re.sub(hashtag_p, "", string)
    string = p.clean(string)
    string = re.sub(r"\bhttp.*\b", "", string)
    string = re.sub(r'[()"#]', "", string)
    string = re.sub(r"\.{2,}", " … ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = " ".join(t for t in string.split() if t not in punctuation)
    return string.strip()


if __name__ == "__main__":
    pass
