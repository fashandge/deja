import re
#from IPython.core.debugger import Tracer
import src.experiment.utilities as util

# the maximum number of words in a sentence
MAX_SENTENCE_WORDS = 18
MAX_TOTAL_WORDS = 30
MAX_TOTAL_CHARS = 150
UPPERCASE_AS_NAMEENTITY = True

paren_patt = re.compile(r'\([^\)]*\)')
paren_patt2 = re.compile(r'\([^\)]{0,7}\)')

'''
(1) Clean title and descriptions and get individual sentences.
Each flickr photo has a title and a description, segment the title and the first line of the description into individual sentences. In my dataset, I keep sentences from both title and description to generate (image, caption) pairs, where each caption is a single sentence.  

Example usage in my dataset:
        # I use a set of 698 nouns (viusal objects) as queries to crawl captions from flickr.com
        # I keep only captions containing the query noun
        query = 'bird' # assume current photo is crawled by querying "bird" at flickr.com

        # title = 'a bird flies in sky (3) http://www.example.com'
        # description = 'This is an interesting bird. blah blah'
        title = preprocess_title(title)
        description = preprocess_description(description)

        title, _ = process_text(title, 'bird', 'n')
        description, _ = process_text(description, 'bird', 'n')
        
        # now title (or descrition) might be an empty string, or one more more sentence concatenated 
         by spaces. Yes, I will segment the title and description into sentences again to generate (image, caption) pairs.
         Call util.segment_sentence(title) to segment a title back into invidual sentences.

(2) Canonicalize individual sentences
    use the function canonical_sentence(sentence, tags)
    you've already done pos tagging (tags is a list of pos tags)

    An example way to organize all of the image-sentence pairs and canonicalize them would look like:
        create_canonical_form(df)
        you don't have to do it this way, just for illustration purpose.

(3) find deja captions whose canonical forms are repeated by at least two images (and from differnet users)
    This is fairly simple processing. The code depends on your data structure.
    One approximation I use is to find deja captions within each group specified by a query term (say 'bird', 'dog', etc), rather than entire dataset. 
    
'''

def preprocess_title(title):
    # remove 'http://' all the way to the end of the title
    http_idx = title.find('http://')
    if http_idx >= 0:
        title = title[0:http_idx]

    # remove parenthesis
    title = paren_patt.sub('', title)

    title = title.strip()
    return title

def preprocess_description(description):
    # take only the first line of the description
    idx = description.find(u'\n')
    if idx >= 0:
        description = description[:idx]

    # todo: no need to remove parenthesis for description?
    description = paren_patt2.sub('', description)
    return description

def process_text(text, query, query_pos):
    ''' process a paragraph

    text: a title or a description
    query: the query word used to crawl the text
    query_pos: pos tag ('v' for verb, 'n' for noun, usually use 'n')


    Return:
        sentences: a list of clean sentences concatenated by spaces
        sentences_ne: a list of clean sentences with name entities (
            identified by Captitalized words), not used in my experiments,
            you can ignore this value, use only "sentences"
    '''
    sentences = util.tokenize_text(text)
    sentences = [remove_tags(sentence) for sentence in sentences]
    sentences = [sentence for sentence in sentences
                 if ((len(sentence) > 1) and
                     (len(sentence) <= MAX_SENTENCE_WORDS) and
                     (not contain_unwanted_words(sentence)) and
                     (contain_query_words(sentence, query,
                                          query_pos=query_pos))
                    )]
    tags = util.tag_sentences(sentences)
    if query_pos == 'v':
        sentences = [sentence
                     for tag, sentence in zip(tags, sentences)
                     if contain_verb(tag, query) and
                     contain_human_or_animal(tag)
                     ]
    else:
        sentences = [sentence
                     for tag, sentence in zip(tags, sentences)
                     if contain_wanted_tags(tag)]

    sentences_ne = [sentence for sentence in sentences
                    if contain_named_entity(sentence)]
    sentences_ne = ensure_length_constraint(sentences_ne)
    sentences = [sentence for sentence in sentences
                    if not contain_named_entity(sentence)]
    sentences = ensure_length_constraint(sentences)

    sentences = [' '.join(sentence) for sentence in sentences]
    sentences_ne = [' '.join(sentence)
                    for sentence in sentences_ne]
    return (u' '.join(sentences), u' '.join(sentences_ne))

def process_photo(photo, query, query_pos, heavy=True):
    ''' clean photo title and description
    '''
    # if heavy
    photo['title'] = preprocess_title(photo['title'])
    photo['description'] = preprocess_description(photo['description'])

    photo_ne = {}
    photo_ne.update(photo)
    if heavy:
        photo['title'], photo_ne['title'] = process_text(
            photo['title'], query, query_pos)
        photo['description'], photo_ne['description'] = process_text(
            photo['description'], query, query_pos)

    if discard(photo):
        photo = None
    if discard(photo_ne):
        photo_ne = None
    return (photo, photo_ne)

def remove_tags(sentence):
    # remove trailing '#''s, and remove '#' in the middle
    count = len(sentence)
    res = []
    i = 0
    while(i < count):
        if sentence[i] == '#':
            if i >= count-2 or sentence[i+2] == '#':
                break
            else:
                i += 1
        else:
            res.append(sentence[i])
            i += 1

    return res

def contain_unwanted_words(tokens):
    for token in tokens:
        if not util.is_ascii(token):
            return True

        lower = token.lower()
        if (lower in ['i', 'my', 'me', 'we', 'our'] or
            lower.endswith('.com')):
            return True

    return False

def contain_named_entity(tokens):
    for token in tokens[1:]:
        if UPPERCASE_AS_NAMEENTITY and token[0].isupper():
            return True
    return False

def contain_wanted_tags(tag):
    ''' whether the pos tags of a sentence contain at least one of VERB, ADJ, or PRP
    '''
    for _, pos in tag:
        if (pos.startswith('VB') or 
            pos.startswith('JJ') or 
            pos == 'IN'):
            return True

    return False

def contain_verb(tag, verb):
    '''whether a sentence contain 'verb' tagged with 'VB'
    '''
    for word, pos in tag:
        if (pos.startswith('VB') and
            util.lemmatize(word, 'v')==verb):
            return True

    return False

def contain_human_or_animal(tag):
    for word, pos in tag:
        if pos.startswith('NN') and util.is_animate(word):
            return True

    return False

def contain_query_words(sentence, query, query_pos='v'):
    # has to compare each query word to the base form of the
    # words in the sentence, which might be time consuming
    if query:
        keywords = query.split()
        for keyword in keywords:
            sentence = [util.lemmatize(word, query_pos) for word
                        in sentence]
            if keyword not in sentence:
                return False

    return True

def ensure_length_constraint(sentences):
    '''keep the first N sentences to meet the constraint of
    total number of words and total chars not exceeding the limits
    '''
    i = 0
    words = 0
    chars = 0
    while i < len(sentences):
        word_count = len(sentences[i])
        words += word_count
        if words > MAX_TOTAL_WORDS:
            break
        # word_count is to count spaces for concatenation
        chars += sum(map(len, sentences[i])) + word_count
        if chars > MAX_TOTAL_CHARS:
            break

        i += 1

    return sentences[0:i]



def discard(photo):
    if(photo['title'] == '' and photo['description']==''):
        return True
    return False

def canonical_tagged_sentence(tokens, tags):
    keep = []
    for token, tag in zip(tokens, tags):
        if tag in ['DT', 'PRP$', 'TO', '.', '$', ':', ',',
                   'POS', '``', "''", 'CD']:
            continue
        token = util.lemmatize(token, util.tag2wnpos(tag))
        if tag == 'IN':
            token = 'IN'
        elif token == "n't":
            token = 'not'
        elif token == "'s" and tag.startswith('VB'):
            token = 'be'
        elif token == "'re":
            token = 'be'
        keep.append(token)

    # use str is because pytable doesn't support unicode
    # in case we want to store it to pytable
    return str(' '.join(keep))

def canonical_sentence(sentence, tags):
    '''
    sentence: a string (a single-sentence caption)
    tags: a list of pos tags of that sentence
    '''
    tokens = sentence.split()
    return canonical_tagged_sentence(tokens, tags)

def create_canonical_form(df):
    ''' Just for illustration purpose:
    assume df is a pandas dataframe (a table) with columns 
    ['img_url', 'sentence']
    '''
    sentences = [sentence.split()
                 for sentence in df.sentence]
    # you can also use NLTK pos tagger, 
    # I used standford parser to redo the pos tagging for better pos tagging quality
    tags = util.stanford_batch_tag(sentences)
    tags = [[pos for _, pos in tag]
            for tag in tags]
    df['tag'] = tags
    print 'creating canonical form of sentences...'
    df['canonical'] = df.apply(
        lambda row: canonical_sentence(
            row['sentence'], row['tag']), 
        axis=1)
    return df
