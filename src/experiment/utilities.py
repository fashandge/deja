import os
import re
import itertools
import time
import datetime
import string
import random
import nltk
import toolz
from nltk.corpus import wordnet as wn
#from nltk.corpus import verbnet as vn
#from nltk.corpus import framenet as fn
from nltk.stem import WordNetLemmatizer
from nltk.tag.stanford import POSTagger
from nltk.corpus import stopwords
#from pattern.en import conjugate
import pandas as pd
import numpy as np
#from IPython.core.debugger import Tracer
#import matlab_wrapper
#from jobman import DD
#import src.turboparser.nlp_pipeline as turbo_pipeline

wn_ic = nltk.corpus.wordnet_ic.ic('ic-brown.dat')

STOP_WORDS = stopwords.words('english')
#STOP_WORDS.extend(
#    ["n't", 'could', 'would', 'seem', 'must'
#    ])+
STOP_WORDS = dict(zip(STOP_WORDS, range(len(STOP_WORDS))))
for word in 'he her him his she'.split():
    del STOP_WORDS[word]

wntype_patt = re.compile(r'^.+\.n\.\d+$')
word_patt = re.compile(r'^[a-zA-Z]+$')
MAX_WORD_LEN = 20
relation_patt = re.compile(r'^[a-zA-Z_]+$')
MAX_RELATION_LEN = 25
PHYSICAL_ENTITY_TYPE  = u'physical_entity.n.01'
fv_patt = re.compile(r'^([a-z]+)_.+$')

ANIMAL_TYPE = 'animal.n.01'
PERSON_TYPE = 'person.n.01'
PEOPLE_TYPE = 'people.n.01'
PLANT_TYPE = 'plant.n.02'
ARTIFACT_TYPE = 'artifact.n.01'
LIVING_THING_TYPE = 'living_thing.n.01'

prps = ['he', 'him', 'she', 'her', 'i', 'we',
        'they', 'me', 'us', 'them']

wnl = WordNetLemmatizer()

matlab = None
def matlab_workspace():
    matlab = matlab_session()
    return matlab.workspace

def matlab_session():
    global matlab
    if matlab is None:
        matlab = matlab_wrapper.MatlabSession()
    return matlab

def reset_matlab_workspace():
    global matlab
    matlab = None
    return matlab_workspace()

def option_iter_plain(opt, included_keys=None):
    keys = opt.keys()
    if included_keys:
        keys = [key for key in keys
                if key in included_keys]
    values = [opt[key] if isinstance(opt[key], list) or
                          isinstance(opt[key], np.ndarray)
              else [opt[key]]
              for key in keys]
    for opt_values in itertools.product(*values):
        yield DD(zip(keys, opt_values))

def option_iter(opt, included_keys=None):
    opt = opt.copy()
    for key in opt:
        if isinstance(opt[key], DD):
            opt[key] = list(option_iter_plain(opt[key]))
    return option_iter_plain(opt)

def arguments(ret_posargs=False):
    """Returns tuple containing dictionary of calling function's
       named arguments and a list of calling function's unnamed
       positional arguments.
    """
    from inspect import getargvalues, stack
    posname, kwname, args = getargvalues(stack()[1][0])[-3:]
    posargs = args.pop(posname, [])
    args.update(args.pop(kwname, []))
    return (args, posargs) if ret_posargs else args

def flatten_args(args):
    ''' flatten arguments into a hash table of key-value pairs
    args itself is a hash obtained by util.arguments(). Some item value
    is a dictionary or DD, other itms are not. We need to merge dicts, and add
    non-dict items as a single dict of arguments. So that we can call make_key
    '''
    dicts = toolz.valfilter(lambda v: isinstance(v, dict), args)
    d = toolz.merge(dicts.values())
    items = toolz.valfilter(lambda v: not isinstance(v, dict), args)
    d.update(items)
    return d

def pick(whitelist, d):
    sub = toolz.keyfilter(
        lambda key: key in whitelist, d)
    if isinstance(d, DD):
        return DD(sub)
    else:
        return sub

def omit(blacklist, d):
    return toolz.keyfilter(
        lambda key: key not in blacklist, d)

def singularize(word):
    from pattern.en import singularize
    return singularize(word)

def stem(word):
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer('english')
    return stemmer.stem(word)

def strsim(string1, string2, method='default'):
    '''
    method = 'default', 'partial'
    '''
    from fuzzywuzzy import fuzz
    if method == 'default':
        return fuzz.ratio(string1, string2)
    elif method == 'partial':
        return fuzz.partial_ratio(string1, string2)
    else:
        raise Exception(
            'method {} not supported'.format(method))

def tokenize_string(string, vocabulary):
    '''tokenize a string that is a concatenation of words
    without spaces into a list of words
    '''
    words = []
    length = len(string)
    start = 0
    while start < length:
        found = False
        for stop in range(length, start, -1):
            word = string[start:stop]
            if word in vocabulary:
                words.append(word)
                found = True
                start = stop
                break
        if not found:
            words.append(string[start:])
            break

    return words

def map(func, iterable, chunksize=1, processes=2):
    if processes == 1:
        import toolz
        return list(toolz.map(func, iterable))
    else:
        import pathos.multiprocessing as mp

        pool = mp.Pool(processes=processes)
        result = pool.map(func, iterable, chunksize=chunksize)
        pool.close()
        pool.join()
        return result

def file_name(path, ext=False):
    name = os.path.basename(path)
    if ext:
        return name
    else:
        return os.path.splitext(name)[0]

def list_files(folder, type='file', patt=None):
    # type = 'file', 'folder', or 'all'
    import glob
    if not patt:
        patt = '*'
    patt = folder + '/' + patt
    paths = glob.glob(patt)
    if type == 'file':
        return filter(os.path.isfile, paths)
    elif type == 'folder':
        return filter(os.path.isdir, paths)
    elif type == 'all':
        return paths
    else:
        raise Exception('type {} not supported'.format(type))

nonascii_patt = re.compile(r'[^\x00-\x7F]+')
def remove_nonascii_chars(string):
    return nonascii_patt.sub('', string)

def is_stopword(word):
    return word.lower() in STOP_WORDS

def image_url(farm, server, image_id, secret):
    return ('http://farm{}.staticflickr.com/'
            '{}/{}_{}.jpg').format(
        farm, server, image_id, secret
    )

url_patt = re.compile(r'http://[^/]+\.com/(\d+)/(\d+)_(\w+)\.jpg$')
def imageurl2key(url):
    m = url_patt.match(url)
    if m:
        return '{image_id}_{secret}_{server}_{image_id}'.format(
            image_id=m.group(2),
            secret=m.group(3),
            server=m.group(1)
        )
    else:
        return None

def format_timestamp(timestamp):
    dt = datetime.datetime.fromtimestamp(timestamp)
    return '{:%Y-%m-%d-%H-%M-%S}'.format(dt)

def is_ascii(word):
    for char in word:
        if char not in string.printable:
            return False

    return True

def segment_sentence(text):
    return nltk.sent_tokenize(text)

def tag2wnpos(tag):
    if tag.startswith('VB'):
        return 'v'
    elif tag.startswith('NN'):
        return 'n'
    elif tag.startswith('JJ'):
        return 'a'
    elif tag == 'RB':
        return 'r'
    else:
        return 'n'

def tokenize_sentence(sentence):
    return nltk.word_tokenize(sentence)

def tokenize_text(text):
    '''tokenize a paragraph into a list of token list for every
    sentence
    '''
    sentences = segment_sentence(text)
    return [nltk.word_tokenize(sentence)
            for sentence in sentences]

def stanford_tag(sentence):
    ''' use stanford tagger to tag a single tokenized sentence
    '''
    import src.experiment.path as path
    tagger = POSTagger(path.stanford_tagger_model_path(),
                       path.stanford_tagger_path(),
                       java_options='-Xmx16g -XX:MaxPermSize=256m')
    return tagger.tag(sentence)

def stanford_batch_tag(sentences):
    '''use stanford tagger to batch tag a list of tokenized
    sentences
    '''
    import src.experiment.path as path
    # need to replace the model path and tagger path of standford parser 
    # in your computer (I use two functions here, you can hard code the paths if 
    # you like)
    tagger = POSTagger(path.stanford_tagger_model_path(),
                       path.stanford_tagger_path())
    return tagger.batch_tag(sentences)

def tag_sentences(sentences):
    ''' use nltk to pos tag a list of tokenized sentences
    '''
    return [nltk.pos_tag(tokens)
            for tokens in sentences]

def timestamp(year, month, day, 
              hour=0, minute=0, second=0):
    return time.mktime(datetime.date(
        year, month, day).timetuple())

def recover_verb_relation(relation):
    ''' recover swim_in to 'swims in'
    '''
    words = relation.split('_')
    if words[0] == lemmatize(words[0]):
        if len(words)>=2 and words[1]=='by':
            # assume "hide by" should be "hidden by"
            # even though this gets some cases wrong (bird stop
            # by tree)
            words[0] = conjugate(words[0], 'ppart')
        else:
            words[0] = conjugate(words[0], tense='present', person=3,
                                 number='singular')
    return ' '.join(words)

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def sentence_frame_ids(verb):
    synset = wn_synset(verb, 'v')
    if synset:
        return synset.frame_ids()
    else:
        return []

def get_subjclue(word, pos):
    def word_type(pos):
        if pos.startswith('VB'):
            return 'verb'
        elif pos.startswith('NN'):
            return 'noun'
        elif pos.startswith('JJ'):
            return 'adj'
        elif pos.startswith('RB'):
            return 'adverb'
        else:
            return 'other'
    
    import src.im2text.data as data
    subjclues = data.subjectivity_clues()
    lemma = lemmatize(word, pos)
    entry = subjclues.get((lemma, word_type(pos)), None)
    if entry:
        return entry
    else:
        entry = subjclues.get((lemma, 'anypos'), None)
        return entry

def sentiment(word, pos):
    entry = get_subjclue(word, pos)
    if entry:
        if entry['type'].startswith('strong'):
            kind = 'strong'
        elif entry['type'].startswith('weak'):
            kind = 'weak'
        return entry.priorpolarity, kind
    else:
        return None, None

def noun_subjectivity(noun):
    import src.im2text.data as data
    subjclues = data.subjectivity_clues()
    entry = subjclues.get((lemmatize(noun, wn.NOUN), 'noun'), None)
    if entry:
        return 'strong' if entry.type.startswith('strong') else 'weak'
    else:
        return None

def verb_subjectivity(verb, synonym_threshold=3):
    import src.im2text.data as data
    subjclues = data.subjectivity_clues()
    entry = subjclues.get((lemmatize(verb, wn.VERB), 'verb'), None)
    if entry is not None:
        return 'strong' if entry['type'].startswith('strong') else 'weak'
    entries = [subjclues.get((synonym, 'verb'), None) 
               for synonym in synonyms(verb)]
    subjs = [None if not entry else entry['type'] for entry in entries]
    # at least 2 synonyms are in subj lexicon, then we consider the word as subjective
    if len(filter(lambda e: e is not None, subjs)) >= synonym_threshold:
        if any(itertools.imap(lambda s: s and s.startswith('strong'), subjs)):
            return 'strong'
        elif any(itertools.imap(lambda s: s and s.startswith('weak'), subjs)):
            return 'weak'
    return None

def is_animate_relation(verb):
    from src.utilities.sentence_frame import SentenceFrame
    import src.experiment.path as path
    sf = SentenceFrame(path.sentence_frame_path())
    return sf.is_animate_relation(verb)

def get_clipboard():
    import clipboard.pyperclip as clip
    return clip.gtkGetClipboard()

def clipdfs():
    import pandas as pd
    from StringIO import StringIO

    buf = StringIO(get_clipboard())
    dfs = pd.read_html(buf, flavor='bs4',
                       header=0, index_col=0)
    dfs = [df[df.columns[1:]] if df.columns[0]=='Unnamed: 0'
           else df
           for df in dfs]
    return dfs

def has_supersense(types, sense):
    if not types:
        return False
    if isinstance(types, dict):
        types = types[types.keys()[0]]
    return any([t.startswith(sense+'.') for t in types])

def abstractness(word, pos=None):
    import src.im2text.data as data
    word = lemmatize(word, pos=pos)
    lexicon = data.abstractness_lexicon()
    if word in lexicon:
        return lexicon[word].A
    else:
        return 1.0

def is_abstract(word, threshold=0.5, pos=None):
    import src.im2text.data as data
    word = lemmatize(word, pos=pos)
    lexicon = data.abstractness_lexicon()
    if word in lexicon:
        return lexicon[word].A >= threshold
    else:
        return False

def is_motion(verb):
    synset = wn_synset(verb, wn.VERB)
    return (synset and synset.lexname() == 'verb.motion')

def random_string(length):
    return ''.join(random.choice(string.ascii_uppercase + string.digits)             for x in range(length))

def normalize_relation(relation):
    '''
    keep prep's the same; trim other relations (usually verbs)
    to the base form of the first verb
    '''
    if is_prep(relation):
        return relation
    else:
        return normalize_verb_relation(relation)

def is_prep(relation):
    return relation.startswith('prep_')

def has_ancestor(obj, wntype):
    types = wntypes(obj)
    if types:
        return (wntype in types[obj])
    else:
        return False

def object_class(obj):
    if is_animal(obj):
        return 'animal'
    elif is_artifact(obj):
        return 'artifact'
    elif is_person(obj):
        return 'person'
    elif is_plant(obj):
        return 'plant'
    elif is_naturalobject(obj):
        return 'naturalobject'
    else:
        return None

def is_naturalobject(obj):
    return ((not obj in prps) and (not is_livingthing(obj)) and
            (not is_artifact(obj)) and
            (not is_person(obj))
    )

def is_animate(obj):
    if obj.lower() == 'men':
        return True
    return (is_animal(obj) or is_person(obj))

def is_livingthing(obj):
    return has_ancestor(obj, LIVING_THING_TYPE)

def is_artifact(obj):
    return has_ancestor(obj, ARTIFACT_TYPE)

def is_plant(obj):
    return has_ancestor(obj, PLANT_TYPE)

def is_plant_part(obj):
    types = wntypes(obj)
    if types:
        return any(wntype.startswith('plant_') for wntype in types[obj])
    else:
        return False

def is_body_part(obj):
    obj = lemmatize(obj)
    types = wntypes(obj)
    if types:
        return any(wntype.startswith('body_part') for wntype in types[obj])
    else:
        return False

def is_animal(obj):
    return has_ancestor(obj, ANIMAL_TYPE)

def is_person(obj):
    if obj in prps:
        return True

    types = wntypes(obj)
    if types:
        return (PERSON_TYPE in types[obj]) or (
            PEOPLE_TYPE in types[obj])
    else:
        return False

def normalize_verb_relation(name):
    name = name.split()[0]
    m = fv_patt.match(name)
    if m:
        verb = m.group(1)
    else:
        verb = name
    return lemmatize(verb)

def is_prep_relation(relation):
    return relation.startswith('prep_')

def lemmatize(word, pos=wn.VERB):
    # ** by default pos is wn.VERB not NOUN !
    word = word.lower()
    # if pos is not available
    if pos is None:
        l1 = wnl.lemmatize(word, 'n')
        if l1 != word:
            return l1
        l2 = wnl.lemmatize(word, 'v')
        return l2

    if pos not in ['v', 'n', 'a', 'r']:
        pos = tag2wnpos(pos)
    return wnl.lemmatize(word.lower(), pos)

def vnlemmas(word):
    classids = vn.classids(word)
    if classids:
        return list(set([lemma for classid in classids for lemma in vn.lemmas(classid)]))
    else:
        return []

def wnsiblings(word, pos=wn.VERB):
    '''
    return the sibling word list
    '''
    s = wn_synset(word, pos)
    h = s.hypernyms()
    if h:
        return list(set([lemma.name for c in h[0].hyponyms()
                         for lemma in c.lemmas]))
    else:
        return []

def wnhyponyms(word, pos=wn.VERB):
    s = wn_synset(word, pos)
    c = s.hyponyms()
    if c:
        return list(set([lemma.name for syn in c
                         for lemma in syn.lemmas]))
    else:
        return []

def is_synonym(w1, w2, pos=wn.NOUN):
    return (w2 in synonyms(w1, pos=pos) or
            w1 in synonyms(w2, pos=pos))

def synonyms(word, pos=wn.VERB):
    synset = wn_synset(word, pos)
    if synset:
        return list(set(lemma.name()
                        for lemma in synset.lemmas()))
    else:
        return []

def wn_synset(word, pos=wn.NOUN):
    synsets = wn.synsets(word, pos)
    if synsets:
        return synsets[0]
    else:
        return None

def is_wntype(obj):
    if wntype_patt.match(obj):
        return True
    else:
        return False

def is_visual_relation(rel):
    ''' discard non-ascii relations '''
    m = relation_patt.match(rel)
    if m:
        return True
    else:
        return False

def is_normal_relation(relation):
    # discard non-ascii relation
    if not relation_patt.match(relation):
        return False

    return len(relation)<=25 and (
        is_prep_relation(relation) or
        is_normal_word(normalize_verb_relation(relation)))

def is_normal_object(obj, types=None):
    return is_normal_word(obj) and (
        is_physical_entity(obj, types))

def is_normal_word(word):
    '''
    whether a word contains only ascii chars and len <= 20
    '''
    return (word_patt.match(word)!=None) and (len(word)<=20)

def is_physical_entity(obj, types=None):
    '''
    args:
        types:  a dict of types for words
    '''
    if obj in prps:
        return True

    obj = lemmatize(obj, 'n')

    #m = word_patt.match(obj)
    #if not m:
    #    return False

    if not types:
        types = wntypes(obj)
    if obj not in types:  # not in wordnet
        return False   # return True
    else:
        return (PHYSICAL_ENTITY_TYPE in types[obj]) or (
            PEOPLE_TYPE in types[obj])

def prptypes(prp):
    man_prps = ['he', 'him']
    woman_prps = ['she', 'her']
    general_prps = ['i', 'we', 'they', 'me', 'us', 'them']

    types = []
    if prp in man_prps:
        synset = wn.synset('man.n.01')
    elif prp in woman_prps:
        synset = wn.synset('woman.n.01')
    elif prp in general_prps:
        synset = wn.synset('person.n.01')

    return hypernyms(synset)

def hypernyms(synset):
    ''' return a list of the names of hypernyms, including the synset itself
    '''
    if isinstance(synset, basestring):
        synset = wn_synset(synset)
    types = [s.name() for path in synset.hypernym_paths()
             for s in path]
    return list(set(types))

def wntypes(*args):
    types = {}
    pos = None
    if args[-1] in ['a', 'v', 'n', 'r']:
        pos = args[-1]
        args = args[:-1]

    def wntype_list(word_list):
        types = {}
        for word in word_list:
            synsets = wn.synsets(word)
            if pos:
                synsets = filter(lambda s: s.pos()==pos, synsets)
            if synsets:
                synset = synsets[0]
                types[word] = hypernyms(synset)
        return types

    for a in args:
        if type(a) == list:
            types.update(wntype_list(a))
        elif type(a) == tuple:
            types.update(wntype_list(list(a)))
        elif isinstance(a, basestring):
            types.update(wntype_list([a]))

    return types

def wnsim(synset1, synset2, method='all'):
    synset_patt = re.compile(r'^.+\..+\.\d+$')

    if synset_patt.match(synset1):
        s1 = wn.synset(synset1)
    else:
        s1 = wn_synset(synset1)

    if synset_patt.match(synset2):
        s2 = wn.synset(synset2)
    else:
        s2 = wn_synset(synset2)

    if s1 is None or s2 is None:
        return 0

    if method == 'lin':
        return wn.lin_similarity(s1, s2, wn_ic)
    elif method == 'res':
        return wn.res_similarity(s1, s2, wn_ic)
    elif method == 'jcn':
        return wn.jcn_similarity(s1, s2, wn_ic)
    elif method == 'wup':
        return wn.wup_similarity(s1, s2)
    elif method == 'path':
        return wn.path_similarity(s1, s2)
    elif method == 'lch':
        return wn.lch_similarity(s1, s2)
    elif method == 'all':
        return [
            ('lin', wn.lin_similarity(s1, s2, wn_ic)),
            ('res', wn.res_similarity(s1, s2, wn_ic)),
            ('jcn', wn.jcn_similarity(s1, s2, wn_ic)),
            ('wup', wn.wup_similarity(s1, s2)),
            ('path', wn.path_similarity(s1, s2)),
            ('lch', wn.lch_similarity(s1, s2))
        ]



class Timer(object):

    def __init__(self, type='wall'):
        self._start = 0
        if type == 'wall':
            self.gettime = time.time
        else:
            self.gettime = time.clock
        self.start()

    def start(self):
        self._start = self.gettime()
        return self._start

    def stop(self, event=''):
        span = self.gettime() - self._start
        import src.experiment.logger as logging
        logging.debug(
            '{} cost time {}',
            event, span)

        self._start = self.gettime()
        return span


def savedf(store, key, df, flatten_list=True, description=None, **kwargs):
    ''' save dataframe to specified path. Replace existing one.
    '''
    if flatten_list:
        list_cols = [col for col in df.columns
                     if isinstance(df.iloc[0][col], list) or 
                        isinstance(df.iloc[0][col], set)]
        if list_cols:
            df = df.copy()
        for col in list_cols:
            df[col] = df[col].apply(lambda li: ' '.join(li))
            df[col] = df[col].apply(str)
    
    def rollback(key, backup):
        if backup is not None:
            remove_node(store, key)
            store.append(key, backup, nan_rep='')
            store.flush()

    backup = None
    if store.get_node(key):
        backup = store[key]
        remove_node(store, key)
    # nan_rep='' is to avoid word 'nan' being treated by
    # pandas as NaN when appending data frame to HDFStore
    try:
        store.append(key, df, nan_rep='', **kwargs)
        store.flush()

        # do verification
        if not all(store[key] == df):
            print ('verification failed '
                   'after save data frame to {} ').format(key)
            rollback(key, backup)
            return False
    except:
        rollback(key, backup)

    if description:
        describe_table(store, key, description)

    return True

def remove_node(store, key):
    try:
        store.remove(key)
        store.flush()
    except KeyError:
        pass

def appenddf(store, key, df, description=None, **kwargs):
    store.append(key, df, **kwargs)
    store.flush()
    if description:
        describe_table(store, key, description)

def describe_table(store, key, description):
    node = store.get_node(key)
    node.table.attrs['description'] = description
    store.flush()

def write_attr(store, key, attr, value):
    attr_table = store.get_storer(key).attrs
    attr_table[attr] = value
    store.flush()

def get_attrs(store, key, attr=None):
    attr_table = store.get_storer(key).attrs
    if not attr:
        return attr_table
    else:
        return attr_table[attr]

def merge(df1, df2, keep_order=True, reset_index=True, **kwargs):
    ''' merge two data frames, keep order according to df1
    '''
    if not keep_order:
        return pd.merge(df1, df2, **kwargs)
    else:
        idx_col = 'index_' + random_string(5)
        df1[idx_col] = range(df1.shape[0])
        kwargs['sort'] = False
        join = pd.merge(df1, df2, **kwargs)
        df1.drop(idx_col, axis=1, inplace=True)
        join.sort(idx_col, inplace=True)
        join.drop(idx_col, axis=1, inplace=True)
        if reset_index:
            join.reset_index(drop=True, inplace=True)
        return join

def sequniq(seq):
    # return unique elements in a sequence, keeping its order
    # it just eliminate neighboring duplciates
    mask = np.diff(seq) != 0
    mask = np.r_[True, mask]
    return seq[mask]

def dump_list(items, file, mode='w'):
    with open(file, mode) as handle:
        for item in items:
            handle.write(str(item) + '\n')

def load_list(file, auto_num=True):
    items = []
    digit_patt = re.compile(r'^(\d|\.)+$')
    with open(file, 'r') as handle:
        for line in handle:
            line = line.strip()
            if auto_num and digit_patt.match(line):
                line = float(line)
            items.append(line)

    return items

def get_options(opt, *keys):
    return tuple(
        [opt.get(key, None) for key in keys]
    )

def swap_file(fsrc, fdst):
    ''' swap two files given their paths
    '''
    import shutil
    ftmp = os.path.dirname(fsrc) + '/' + random_string(10)
    shutil.move(fsrc, ftmp)
    shutil.move(fdst, fsrc)
    shutil.move(ftmp, fdst)

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def format_df(df, float_precision=3):
    float_format = '{{:.{}f}}'.format(float_precision)
    return df.to_string(
        float_format=lambda n: float_format.format(n))

def poll(event_ready, interval, timeout):
    start = time.time()
    while time.time()-start <= timeout:
        if event_ready():
            return True
        time.sleep(interval)
    return False 

import nltk.corpus as corpus
US_CITIES = corpus.gazetteers.words('uscities.txt')
US_STATES = corpus.gazetteers.words('usstates.txt') + \
                corpus.gazetteers.words('usstateabbrev.txt')
COUNTRIES = corpus.gazetteers.words()
GEO_LOCATIONS = US_CITIES + US_STATES + COUNTRIES
GEO_PATTERNS = [
    re.compile(r'\b{}\b'.format(location.lower()))
    for location in GEO_LOCATIONS
    if is_ascii(location)
]
def has_geolocation(sentence):
    sentence = sentence.lower()
    for pattern in GEO_PATTERNS:
        if pattern.search(sentence):
            return True
    return False

def ascii_df(df):
    for col in df.columns:
        if isinstance(df[col].iloc[0], basestring):
            df[col] = df[col].apply(str)
    return df

def extract_dependency(text):
    lines = filter(lambda content: content != '', text.split('\n'))
    d = DD()
    d.words = []
    d.tags = []
    d.heads = []
    d.deps = []
    for line in lines:
        fields = line.split()
        d.words.append(fields[1])
        d.tags.append(fields[3])
        d.heads.append(int(fields[-2]) - 1)
        d.deps.append(fields[-1])
    return d

def print_deps(d):
    for i, (word, tag, head, dep) in enumerate(
            zip(d.words, d.tags, d.heads, d.deps)):
        print '\t'.join((str(i), word, tag, str(head), dep))
    print
    print

def turbo_parse(sentence, verbose=True):
    pipe = turbo_pipeline.get_turbo_pipeline()
    text = pipe.parse_conll(sentence, 'EN')
    d = extract_dependency(text)
    if verbose:
        print_deps(d)
    return d

def amt_price(n_task, hit_price=0.05, tasks_per_hit=7, n_workers=3):
    return n_task * 1.0 / tasks_per_hit * n_workers * hit_price

def sample_df(df, n_sample):
    ''' randomly sample n_sample rows and return a sub df
    '''
    if n_sample > df.shape[0]:
        n_sample = df.shape[0]
    return df.iloc[
        np.random.choice(range(df.shape[0]), n_sample, replace=False)
    ]


def kappa(y_true, y_pred, type='cohens'):
    import statsmodels.stats.inter_rater as irater
    yy = (y_true & y_pred).sum()
    yn = (y_true & (~y_pred)).sum()
    nn = ((~y_true) & (~y_pred)).sum()
    ny = ((~y_true) & (y_pred)).sum()
    result = np.array([[yy, yn], [ny, nn]])
    if type == 'cohens':
        stat = irater.cohens_kappa(result)
        score = stat['kappa']
    elif type == 'fleiss':
        score = irater.fleiss_kappa(result)
    return score, result

    # my implementation
    #yy = (y_true & y_pred).sum()
    #yn = (y_true & (~y_pred)).sum()
    #nn = ((~y_true) & (~y_pred)).sum()
    #ny = ((~y_true) & (y_pred)).sum()
    #result = np.array([[yy, yn], [ny, nn]])
    #total = result.sum()
    #pa = (yy + nn) * 1.0 / total
    #pe_yes = result[0].sum() * 1.0 / total * \
    #         result[:, 0].sum() * 1.0 / total
    #pe_no = result[1].sum() * 1.0 / total * \
    #        result[:, 1].sum() * 1.0 / total
    #pe = pe_yes + pe_no
    #k = (pa - pe) / (1 - pe)

def process_df_by_sections(df, n_section, func, tmp_output=None, ignore_index=True):
    import src.experiment.logger as logging
    size = df.shape[0]
    section_size = size / n_section + (size%n_section > 0)
    results = []
    n_success = 0
    for i, section in enumerate(chunks(range(size), 
                                            section_size)):
        try:
            logging.debug('processing section {} of {}...', i, n_section)
            sub = df.iloc[section]
            sub = func(sub)
            results.append(sub)
            if tmp_output is not None:
                tmp_output(i, sub)
            n_success += 1
        except Exception as e:
            logging.debug('exception when processing section {}'.format(i))
            logging.debug('jump this section for now and remember to come back to it'.format(i))
            logging.debug('{}', e)

    logging.debug('concatenating results...')
    results = pd.concat(results, ignore_index=ignore_index)
    return results

def process_data_by_sections(data, n_section, func, tmp_output=None, 
                             merge_results=True, ignore_index=True):
    import src.experiment.logger as logging
    size = len(data)
    section_size = size / n_section + (size%n_section > 0)
    if merge_results:
        results = []
    n_success = 0
    def sub_section(data, section):
        if isinstance(data, pd.DataFrame):
            return data.iloc[section]
        else:
            return data[section]

    for i, section in enumerate(chunks(range(size), 
                                            section_size)):
        try:
            logging.debug('processing section {} of {}...', i, n_section)
            sub = sub_section(data, section)
            sub = func(sub)
            if merge_results:
                results.append(sub)
            if tmp_output is not None:
                tmp_output(i, sub)
            n_success += 1
        except Exception as e:
            logging.debug('exception when processing section {}'.format(i))
            logging.debug('jump this section for now and remember to come back to it'.format(i))
            logging.debug('{}', e)

    logging.debug('finished processing {} sections, {} successes, remember to process skpped sections.',
                  n_section, n_success)
    if merge_results:
        logging.debug('concatenating results...')
        results = pd.concat(results, ignore_index=ignore_index)
        return results
    else:
        return n_section == n_success

def freq2df(freq, columns, ascending=False):
    freq = pd.DataFrame(freq.items(), columns=columns)
    freq.sort(columns[1], ascending=ascending, inplace=True)
    return freq
