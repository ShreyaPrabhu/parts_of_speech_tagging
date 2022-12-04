# parts_of_speech_tagging
Implementation of Viterbi Algorithm for Parts of Speech Tagging. Code is generalized and can be used for any natural language.


## Dataset 

The corpus has been adapted from the Italian (ISDT) and Japanese (GSD) sections of the universal dependencies corpus. The source corpora, documentation, and credits can be found at http://universaldependencies.org

1. Two files (one Italian, one Japanese) with tagged training data in the word/TAG format, with words separated by spaces and each sentence on a new line.
2. Two files (one Italian, one Japanese) with untagged development data, with words separated by spaces and each sentence on a new line.
3. Two files (one Italian, one Japanese) with tagged development data in the word/TAG format, with words separated by spaces and each sentence on a new line, to serve as an answer key.

## Evaluation

The final code was tested on 3 languages - Italian, Japanese and Urdu. The hmm model learnt on training and development data and tested on unseen test data for each of the languages.

| Language  | Total words | Correctly tagged words | Accuracy|
|--------|-------------|-------------|-------------|
| Italian  | 10417  | 9905 | 95.08 |
| Japanese | 12438 | 11497 | 92.43|
| Urdu  |  14806 | 12815 | 86.55|


## Execution

Steps to execute

```python
# The learning program will be invoked in the following way:
python3 hmmlearn.py /path/to/input

# The tagging program will be invoked in the following way:
python3 hmmdecode.py /path/to/input
```
