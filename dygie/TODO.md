1. [] Add NER checks to coref (same) and relation (different).
2. [x] Split NER and linking into two components (Independent - p(E)*p(L) and conditional - p(E)*p(L|E))
3. [] Implement sliding window
4. [] Implement paragraph pairs
5. [x] Remove Material that is False / Reduce class weight
6. [x] Implement random batches
7. [] Implement per-entity type coreference model

NER
===

1. [] Features Input (POS)
2. [] Viterbi Thresholding
3. [] CRF on spans

Coreference
===========

1. [x] Distance Features between Entities
2. [] Contextualised Entities Representation
3. [] Global Entity Features (Co-occurence of words in whole corpus)
4. [] Within the end to end model

Relations
=========

1. [] Find some way to incorporate features from sciERC
2. [] Direct Supervision Annotation
3. [x] Better section information
4. [] Features from Tables
5. [] Teacher Forcing

Presentation
=============

1. [] Explain ternary relation construction
2. [] LSTM over what ?
3. [] Make Material -> Dataset 
4. [] 