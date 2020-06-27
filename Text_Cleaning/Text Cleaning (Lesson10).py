#!/usr/bin/env python
# coding: utf-8

# ## Bag of Words in `sklearn`

# In[1]:


import sys
sys.path.append("/home/cit5/Downloads/ud120-projects-master/tools/")
sys.path.append('/home/cit5/Downloads/ud120-projects-master/choose_your_own')
sys.path.append('/home/cit5/Downloads/ud120-projects-master/datasets_questions')


import os
os.chdir('/home/cit5/Downloads/ud120-projects-master/text_learning')


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

string1 = '''Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec pharetra ornare lacinia. 
Cras et facilisis sem. Mauris fermentum nec dui at aliquam. Mauris ante justo, aliquam 
eget facilisis ac, elementum eget dolor. Nulla lacinia ac lorem vel tempus. Vivamus condimentum 
vulputate semper. Nullam dictum, mauris dignissim auctor suscipit, dui turpis elementum turpis, 
et tempor urna diam sed nisl. Vestibulum mollis quam at nisl egestas tincidunt. Nullam ut nibh 
fringilla, ultricies neque vel, tempor erat. Phasellus vel sem vitae ligula viverra luctus hendrerit 
ac quam. Vivamus dignissim, diam sed porttitor lacinia, elit erat lobortis tellus, nec euismod dolor turpis a leo.'''

string2 = '''Mauris felis nibh, tempor ac pulvinar sed, feugiat at neque. In enim elit, venenatis nec magna eu, vestibulum 
lobortis libero. Nullam scelerisque pulvinar ex consectetur tempor. Morbi congue quis leo auctor facilisis. Mauris 
et diam ultricies, interdum nisi ac, condimentum turpis. Donec a risus sed dolor blandit ultrices. Nulla ac ultrices 
elit. Nulla dictum metus tortor, in sollicitudin enim rutrum id. Nulla magna felis, molestie vel odio et, congue ornare 
nisi. Nunc molestie, mi non blandit sodales, neque diam varius quam, nec tempus metus neque vel nibh. Morbi sit amet 
sapien nec ipsum feugiat lacinia ut eu orci. Mauris id enim tincidunt, aliquet augue quis, vehicula libero. Donec eu 
laoreet nibh.'''

string3 = '''Donec urna massa, faucibus et interdum id, facilisis interdum augue. Nunc enim nulla, tristique sit amet velit 
in, lacinia mollis mi. Nullam malesuada felis sed libero porttitor dapibus ut quis dolor. Curabitur vitae neque 
at arcu condimentum suscipit. Quisque sollicitudin est a elit sollicitudin, ac consectetur nisi blandit. Vestibulum 
rhoncus viverra orci, et porta purus maximus quis. Pellentesque consectetur velit eget orci rutrum mollis. Nullam 
condimentum vehicula ante at dignissim. Curabitur gravida, lorem in vehicula gravida, enim arcu semper nulla, 
id tempor magna eros vel tortor.'''

email_list = [string1, string2, string3]

bag_of_words = vectorizer.fit_transform(email_list)

print vectorizer.vocabulary_.get('id')


# ## Getting Stopwords from NLTK

# In[2]:


from nltk.corpus import stopwords
sw = stopwords.words('english')
print 'Number of stopwords: {0}'.format(len(sw))


# ## Stemming with `nltk`

# In[3]:


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
print stemmer.stem('responsiveness')
print stemmer.stem('responsivity')
print stemmer.stem('unresponsive')


# ## Warming Up with `parseOutText()`

# In[4]:


import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(str.maketrans("", "", string.punctuation))

        words = text_string
        
    return words
    

ff = open("../text_learning/test_email.txt", "r")
text = parseOutText(ff)
print text


# ## Deploying Stemming

# In[5]:


from nltk.stem.snowball import SnowballStemmer

def parseOutText(f):
    '''
    Input: a file containing text
    
    Output: the stemmed words in the input text, all separated by a single space
    '''
    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    
    # the stemmer
    stemmer = SnowballStemmer('english')
    
    # the string of words
    words = ""
    
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(str.maketrans("", "", string.punctuation))

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        for word in text_string.split():
            # stem the word and add it to words
            words += stemmer.stem(word) + ' '       
        
    return words[:-1]
    

ff = open("../text_learning/test_email.txt", "r")
text = parseOutText(ff)
print text


# ## Clean Away "Signature Words"

# In[6]:


import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""

sw = ["sara", "shackleton", "chris", "germani"]
with open("from_sara.txt", "r") as from_sara, open("from_chris.txt", "r") as from_chris:

    from_data = []
    word_data = []

    ### temp_counter is a way to speed up the development--there are
    ### thousands of emails from Sara and Chris, so running over all of them
    ### can take a long time
    ### temp_counter helps you only look at the first 200 emails in the list so you
    ### can iterate your modifications quicker
    temp_counter = 0


    for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
        for path in from_person:
            ### only look at first 200 emails when developing
            ### once everything is working, remove this line to run over full dataset
            
            #temp_counter += 1
            if temp_counter < 200:
                path = os.path.join('..', path[:-1])

                with open(path, 'r') as email:
                    ### use parseOutText to extract the text from the opened email
                    text = parseOutText(email)

                    ### use str.replace() to remove any instances of the words
                    ### ["sara", "shackleton ", "chris", "germani"]
                    for word in sw:
                        if(word in text):
                            text = text.replace(word, "")

                    ### append the text to word_data
                    word_data.append(text.replace('\n',' ').strip())

                    ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
                    if name=='sara':
                        from_data.append(0)
                    else:
                        from_data.append(1)

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )

print word_data[152]

# word data is a list of strings that we get by stemming the emails


# ## `TfIdf` it

# In[7]:


### in Part 4, do TfIdf vectorization here
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
vectorizer.fit_transform(word_data)

feature_names = vectorizer.get_feature_names()

print 'Number of different words: {0}'.format(len(feature_names))


# ## Accessing `TfIdf` features

# In[8]:


print feature_names[34597]

