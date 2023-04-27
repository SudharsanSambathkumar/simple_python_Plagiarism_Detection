# simple_python_Plagiarism_Detection
 NLP|Streamlit
 
To run the application, use the command `streamlit run stream.py` in terminal where this file locates.

This code is a Python program for detecting plagiarism in text documents. It uses several Natural Language Processing (NLP) techniques to compare two text files and determine their similarity. The program consists of several modules for language, spelling, grammar, and sentence analysis.

The program uses the following libraries:
- `nltk`: Natural Language Toolkit for NLP tasks such as tokenization, stopword removal, and sentence segmentation.
- `streamlit`: A framework for creating web-based applications.
- `matplotlib`: A plotting library for visualizing data.
- `langdetect`: A language detection library for identifying the language of the text.
- `spellchecker`: A library for detecting and correcting spelling errors.
- `language_tool_python`: A grammar checker library for identifying grammar mistakes.

The program starts by uploading two text files: an original file and a potentially plagiarized file. It then reads the contents of the files and detects their language using the `langdetect` library.

Next, it checks the spelling of the words using the `spellchecker` library. If any misspelled words are found, it displays them; otherwise, it displays a message that no misspelled words were found.

The program then checks the grammar of the documents using the `language_tool_python` library. It displays the grammar mistakes found in each document, or a message that no grammar mistakes were found.

The program then performs NLP preprocessing on the documents. It tokenizes the documents into words, removes stop words and punctuation, and converts all words to lowercase. It then calculates the Jaccard similarity coefficient and containment measure for the documents, which are measures of their similarity.

Finally, it calculates the longest common subsequence (LCS) between the sentences of the documents. The program computes the maximum length of LCS for a sentence in the suspicious text and sums up all the LCS lengths. It then displays the maximum LCS length and the sum of all LCS lengths.

The results of the analysis are displayed using the `streamlit` framework, which creates a web-based interface for the user to interact with the program.
