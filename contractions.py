# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 22:09:38 2022

@author: angel
"""

contractions = {"i'm": "i am", "i'm'a": "i am about to", "i'm'o": "i am going to", "i've": "i have", 
                "i'll": "i will", "i'll've": "i will have", "i'd": "i would", "i'd've": "i would have", 
                "whatcha": "what are you", "amn't": "am not", "ain't": "are not", "aren't": "are not", 
                "'cause": "because", "can't": "cannot", "can't've": "cannot have", "could've": "could have", 
                "couldn't": "could not", "couldn't've": "could not have", "daren't": "dare not", 
                "daresn't": "dare not", "dasn't": "dare not", "didn't": "did not", "didn’t": "did not", 
                "don't": "do not", "don’t": "do not", "doesn't": "does not", "e'er": "ever", 
                "everyone's": "everyone is", "finna": "fixing to", "gimme": "give me", "gon't": "go not", 
                "gonna": "going to", "gotta": "got to", "hadn't": "had not", "hadn't've": "had not have", 
                "hasn't": "has not", "haven't": "have not", "he've": "he have", "he's": "he is", 
                "he'll": "he will", "he'll've": "he will have", "he'd": "he would", "he'd've": "he would have", 
                "here's": "here is", "how're": "how are", "how'd": "how did", "how'd'y": "how do you", 
                "how's": "how is", "how'll": "how will", "isn't": "is not", "it's": "it is", "'tis": "it is", 
                "'twas": "it was", "it'll": "it will", "it'll've": "it will have", "it'd": "it would", 
                "it'd've": "it would have", "kinda": "kind of", "let's": "let us", "luv": "love", "ma'am": "madam", 
                "may've": "may have", "mayn't": "may not", "might've": "might have", "mightn't": "might not", 
                "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", 
                "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", 
                "ne'er": "never", "o'": "of", "o'clock": "of the clock", "ol'": "old", "oughtn't": "ought not", 
                "oughtn't've": "ought not have", "o'er": "over", "shan't": "shall not", "sha'n't": "shall not", 
                "shalln't": "shall not", "shan't've": "shall not have", "she's": "she is", "she'll": "she will", 
                "she'd": "she would", "she'd've": "she would have", "should've": "should have", "shouldn't": "should not", 
                "shouldn't've": "should not have", "so've": "so have", "so's": "so is", "somebody's": "somebody is", 
                "someone's": "someone is", "something's": "something is", "sux": "sucks", "that're": "that are", 
                "that's": "that is", "that'll": "that will", "that'd": "that would", "that'd've": "that would have", 
                "em": "them", "there're": "there are", "there's": "there is", "there'll": "there will",
                "there'd": "there would", "there'd've": "there would have", "these're": "these are", 
                "they're": "they are", "they've": "they have", "they'll": "they will", "they'll've": "they will have", 
                "they'd": "they would", "they'd've": "they would have", "this's": "this is", "this'll": "this will", 
                "this'd": "this would", "those're": "those are", "to've": "to have", "wanna": "want to", 
                "wasn't": "was not", "we're": "we are", "we've": "we have", "we'll": "we will", "we'll've": "we will have", 
                "we'd": "we would", "we'd've": "we would have", "weren't": "were not", "what're": "what are", 
                "what'd": "what did", "what've": "what have", "what's": "what is", "what'll": "what will", 
                "what'll've": "what will have", "when've": "when have", "when's": "when is", "where're": "where are", 
                "where'd": "where did", "where've": "where have", "where's": "where is", "which's": "which is", 
                "who're": "who are", "who've": "who have", "who's": "who is", "who'll": "who will", 
                "who'll've": "who will have", "who'd": "who would", "who'd've": "who would have", "why're": "why are", 
                "why'd": "why did", "why've": "why have", "why's": "why is", "will've": "will have", "won't": "will not", 
                "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", 
                "y'all": "you all", "y'all're": "you all are", "y'all've": "you all have", "y'all'd": "you all would", 
                "y'all'd've": "you all would have", "you're": "you are", "you've": "you have", "you'll've": "you shall have", 
                "you'll": "you will", "you'd": "you would", "you'd've": "you would have", "'cause": "because", "'d": " would", 
                "'ll": " will", "'re": " are", "'em": " them", "doin'": "doing", "goin'": "going", "nothin'": "nothing", 
                "somethin'": "something", "havin'": "having", "lovin'": "loving", "'coz": "because", "thats": "that is", 
                "whats": "what is", "'aight": "alright", "abt": "about", "acct": "account", "altho": "although", 
                "asap": "as soon as possible", "avg": "average", "b4": "before", "bc": "because", "bday": "birthday", 
                "btw": "by the way", "convo": "conversation", "cya": "see ya", "diff": "different", "dunno": "do not know", 
                "g'day": "good day", "gimme": "give me", "gonna": "going to", "gotta": "got to", "howdy": "how do you do", 
                "idk": "I do not know", "ima": "I am going to", "imma": "I am going to", "innit": "is it not", 
                "iunno": "I do not know", "kk": "okay", "lemme": "let me", "msg": "message", "nvm": "nevermind", 
                "ofc": "of course", "ppl": "people", "prolly": "probably", "pymnt": "payment", "r ": "are ", "rlly": "really", 
                "rly": "really", "rn": "right now", "spk": "spoke", "tbh": "to be honest", "tho": "though", "thx": "thanks", 
                "tlked": "talked", "tmmw": "tomorrow", "tmr": "tomorrow", "tmrw": "tomorrow", "u": "you", "ur": "you are", 
                "n": "and", "wanna": "want to", "woulda": "would have" }