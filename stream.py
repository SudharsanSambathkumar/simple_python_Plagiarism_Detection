import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import streamlit as st
import matplotlib.pyplot as plt

st.title("Plagiarism Detection")
st.subheader("Text Editor/Reader Module:")
# file uploader
orig = st.file_uploader("Select the original file", type=['txt'])
plag = st.file_uploader("Select the plagiarised file", type=['txt'])

if orig and plag:
    # read input files
    orig_contents = orig.read().decode('utf-8', errors='ignore')
    plag_contents = plag.read().decode('utf-8', errors='ignore')
    st.subheader("Language Check Module:")
    from langdetect import detect

    text1 =orig_contents
    language1 = detect(text1)
    st.write('original text is ',language1)
    text2 =plag_contents
    language2 = detect(text2)
    st.write('plag text is ',language2)
    st.subheader("Spelling Check Module:")
    from spellchecker import SpellChecker
    spell = SpellChecker()
    words1 = text1.split()
    misspelled1 = spell.unknown(words1)
    words2 = text2.split()
    misspelled2 = spell.unknown(words2)
    if misspelled1 or misspelled2:
        st.write("it checks the words in common dictionary if your words are in regional language omit it.")
        st.write(misspelled1)
        st.write(misspelled2)
    else:
        st.write("The given document doesn't contains mistake words")
    st.subheader("Grammar Check Module:")
    import language_tool_python

    tool = language_tool_python.LanguageTool('en-US')

    textg2 = text2
    matches2 = tool.check(textg2)
    textg1 = text1
    matches1 = tool.check(textg1)
    if matches1 or matches2:
        for match1 in matches1:
            st.write('\ngrammer mistakes in original document',match1)
        for match2 in matches2:
            st.write('\ngrammer mistakes in testing document',match2)
    else:
        st.write('No grammer mistakes found')
    st.subheader("Sentences Analysis Module:")
    # word tokenisation
    tokens_o = word_tokenize(orig_contents)
    tokens_p = word_tokenize(plag_contents)
    st.write("tokens from original document",tokens_o)
    st.write("tokens from texting document",tokens_p)

    # lowerCase
    tokens_o = [token.lower() for token in tokens_o]
    tokens_p = [token.lower() for token in tokens_p]

    # stop word removal
    # punctuation removal
    st.subheader("NLP Preprocessing Module:")
    stop_words = set(stopwords.words('english'))
    st.write("stopwords are",stop_words)

    punctuations = ['"', '.', '(', ')', ',', '?', ';', ':', "''", '``', '-']
    filtered_tokens_o = [w for w in tokens_o if not w in stop_words and not w in punctuations]

    filtered_tokens_p = [w for w in tokens_p if not w in stop_words and not w in punctuations]

    # Trigram Similiarity measures
    trigrams_o = []
    for i in range(len(tokens_o) - 2):
        t = (tokens_o[i], tokens_o[i + 1], tokens_o[i + 2])
        trigrams_o.append(t)

    s = 0
    trigrams_p = []
    for i in range(len(tokens_p) - 2):
        t = (tokens_p[i], tokens_p[i + 1], tokens_p[i + 2])
        trigrams_p.append(t)
        if t in trigrams_o:
            s += 1

    # jaccord coefficient = (S(o)^S(p))/(S(o) U S(p))
    J = s / (len(trigrams_o) + len(trigrams_p))
    st.write("jaccord coefficient:",J)

    # containment measure
    C = s / len(trigrams_p)
    st.write("containment measure:",C)

    # longest common subsequence
    # dynamic programming algorithm for finding lcs
    def lcs(l1, l2):
        s1 = word_tokenize(l1)
        s2 = word_tokenize(l2)
        # storing the dp values
        dp = [[None] * (len(s1) + 1) for i in range(len(s2) + 1)]

        for i in range(len(s2) + 1):
            for j in range(len(s1) + 1):
                if i == 0 or j == 0:
                    dp[i][j] = 0
                elif s2[i - 1] == s1[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[len(s2)][len(s1)]

    sent_o = sent_tokenize(orig_contents)
    sent_p = sent_tokenize(plag_contents)
    #maximum length of LCS for a sentence in suspicious text
    max_lcs=0
    sum_lcs=0

    for i in sent_p:
        for j in sent_o:
            l=lcs(i,j)
            max_lcs=max(max_lcs,l)
        sum_lcs+=max_lcs
        max_lcs=0

    score=sum_lcs/len(tokens_p)
    st.write("score",score)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    def plot_accuracy(jaccard, containment, score):
        plt.plot([jaccard, containment, score])
        plt.xlabel('Measure')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Plot')
        plt.xticks([0, 1, 2], ['Jaccard', 'Containment', 'Score'])
        st.pyplot()

    plot_accuracy(J,C,score)
