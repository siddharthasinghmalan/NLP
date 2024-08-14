import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer


'''
POS tag list:   
CC: Coordinating conjunction

CD: Cardinal number

DT: Determiner

EX: Existential there

FW: Foreign word

IN: Preposition or subordinating conjunction

JJ: Adjective

VP: Verb Phrase

JJR: Adjective, comparative

JJS: Adjective, superlative

LS: List item marker

MD: Modal

NN: Noun, singular or mass

NNS: Noun, plural

PP: Preposition Phrase

NNP: Proper noun, singular Phrase

NNPS: Proper noun, plural

PDT: Pre determiner

POS: Possessive ending

PRP: Personal pronoun Phrase

PRP: Possessive pronoun Phrase

RB: Adverb

RBR: Adverb, comparative

RBS: Adverb, superlative

RP: Particle

S: Simple declarative clause

SBAR: Clause introduced by a (possibly empty) subordinating conjunction

SBARQ: Direct question introduced by a wh-word or a wh-phrase.

SINV: Inverted declarative sentence, i.e. one in which the subject follows the tensed verb or modal.

SQ: Inverted yes/no question, or main clause of a wh-question, following the wh-phrase in SBARQ.

SYM: Symbol

VBD: Verb, past tense

VBG: Verb, gerund or present participle

VBN: Verb, past participle

VBP: Verb, non-3rd person singular present

VBZ: Verb, 3rd person singular present

WDT: Wh-determiner

WP: Wh-pronoun

WP: Possessive wh-pronoun

WRB: Wh-adverb'''

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(sample_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content() :
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            # TAke a look at curly brackets as they are inverted
            chunkGram = r""" Chunk : {<.*>+}
                                        }<VB.?| IN|DT >+{"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
             
            chunked.draw()

    except Exception as e :
        print(str(e))

process_content()