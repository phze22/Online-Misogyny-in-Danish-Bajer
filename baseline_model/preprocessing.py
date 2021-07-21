import re
import string
from functools import reduce
# from wordsegment import load, segment - just for English available
import demoji
class Preprocessing():

    def __init__(self, post):
        self.post = post

    """ replace empjis with text, i.e. "🎅🏾": "Santa Claus: medium-dark skin tone" """
    def convert_emojis(self):
        emo_dict = demoji.findall(self.post)
        text = self.post
        for key in emo_dict.keys():
            new_string = emo_dict[key]
            text = text.replace(key, new_string)
        self.post = text

    """mark: use of capital letters"""
    def convert_capital_letters(self):
        def _has_cap(token):
            return token.lower() != token and token.upper() != token

        def _all_cap(token):
            return token.lower() != token and token.upper() == token

        exceptions = ['@USER', 'HTTP']

        tokens = self.post.split()
        tokens = ['<has_cap> ' + t if _has_cap(t) and t not in exceptions else t for t in tokens]
        tokens = ['<all_cap> ' + t if _all_cap(t) and t not in exceptions else t for t in tokens]

        self.post = ' '.join(tokens)

    # """censorship detection?"""
    # """observation 1 by annotators: f*** = fuck"""
    # """observation 2 by annotators: f... = fuck"""

    # "spelling mistakes?"
    # def correct_spelling_mistakes(sent):
    #    return sent

    """split hashtags into single words"""
    #def replace_hashtags(sent):
    #    return s'<hashtag> ' + re.sub('#[\S]+', lambda match:  + ' </hashtag>'

    """reduce pattern occurences"""
    def reduce_pattern(self, sent, pattern, keep_num):
            if pattern in string.punctuation:
                re_pattern = re.escape(pattern)
            else:
                re_pattern = f'(({pattern})[\s]*)'
                pattern = pattern + ' '
            pattern_regex = re_pattern + '{' + str(keep_num + 1) + ',}'
            return re.sub(pattern_regex, lambda match: pattern * keep_num, sent)

    """convert urls to not learn from words in the weblink"""
    def convert_urls(self):
        text = self.post.replace(' https\*', 'HTTP')
        self.post = self.reduce_pattern(text, 'HTTP',1)

    """limiting punctuation"""
    def limit_punctuations(self):
        keep_num = 3
        puncs = ['!', '?', '.']
        sent_new = self.post
        for p in puncs:
            sent_new = self.reduce_pattern(sent_new,p, keep_num)
        self.post = sent_new

    """and limit User-mentions"""
    def limit_user_mention(self):
        keep_num = 1
        self.post = self.reduce_pattern(self.post,'@USER', keep_num)

    def get_post(self):
        return self.post

    # compose data
    def compose(*funcs):
        """" Compose functions so that they are applied in chain. """
        return reduce(lambda f, g: lambda x: f(g(x)), funcs[::-1])

def preprocess(data):
    demoji.download_codes()

    #[::-1] reverse, equivalent to [:-len(a)-1:-1] -> slice it reverse up

    #test
    #print(Preprocessing.convert_emojis(list_of_posts[0]))

    preprocessed = []
    for post in data['text'].tolist():
        proc = Preprocessing(post)
        proc.convert_emojis()
        proc.convert_capital_letters()
        # proc.convert_urls()
        proc.limit_punctuations()
        proc.limit_user_mention()
        preprocessed += [proc.get_post()]

    assert(len(preprocessed)==len(data['text']))

    data['text'] = preprocessed
    #preprocessed = [reduce(lambda x, f: f(x), preprocess_funcs, post) for post in list_of_posts]
    return data




