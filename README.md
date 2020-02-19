# Cranfield Corpus Vector Space Model

This project was done as a homework assignment for CS582: Information Retrieval course at the University of Illinois at Chicago during the Spring 2020 term.

----

The [dataset](cranfieldDocs "Cranfield collection") used was the Cranfield collection which is a standard Information Retrieval text collection, consisting of 1400 documents from the aerodynamics field.

Each document in the collection are in SGML format. So the SGML tags (e.g.,\<TITLE>, \<DOC>,\<TEXT>, etc.) have been eliminated and only the text between the  \<TITLE> and \<TEXT> tags have been retained for building the Vector Space Retrieval Model.

For the tasks in this assignment the pre-processing tools implemented in [this project](https://github.com/samujjwaal/CiteSeer-Text-Processing "CiteSeer Collection Text Processing") were reused. The documents after cleaning were stored in a new [folder](preprocessed_cranfieldDocs).

There is also a [list of queries](queries.txt) and a [list of relevant documents](relevance.txt) for each query provided, to calculate the precision and recall values of the retrieval model. The evaluation metrics are saved in an [output](output.txt) file.

The same text pre-processing operations were applied on both the corpus documents and the queries.

---

The tasks in the [assignment](Tasks.pdf "Assignment description") included:

1. Implement an indexing scheme based on the vector space model. *TF-IDF* used for weighting scheme.
2. For each of the queries, determine a ranked list of documents, in descending order of their cosine similarity with the queries.
3. Determine the average precision and recall for the queries, using top 10, 50, 100 and 500 documents in the ranking.

Check out the [Jupyter Notebook](citeseer.ipynb "CiteSeer Collection Text Processing") to see the python implementation of the tasks.

----

**Results**

On evaluating the queries,

Top 10 documents in rank list
Query: 1	Pr: 0.0 	Re: 0.0
Query: 2	Pr: 0.2 	Re: 0.13333333333333333
Query: 3	Pr: 0.2 	Re: 0.13333333333333333
Query: 4	Pr: 0.1 	Re: 0.05555555555555555
Query: 5	Pr: 0.1 	Re: 0.05263157894736842
Query: 6	Pr: 0.4 	Re: 0.2222222222222222
Query: 7	Pr: 0.6 	Re: 0.6666666666666666
Query: 8	Pr: 0.2 	Re: 0.5
Query: 9	Pr: 0.1 	Re: 0.125
Query: 10	Pr: 0.2 	Re: 0.08333333333333333


Avg precision:0.21
Avg Recall:0.2
Top 50 documents in rank list
Query: 1	Pr: 0.0	Re: 0.0
Query: 2	Pr: 0.12	Re: 0.4
Query: 3	Pr: 0.12	Re: 0.4
Query: 4	Pr: 0.06	Re: 0.16666666666666666
Query: 5	Pr: 0.16	Re: 0.42105263157894735
Query: 6	Pr: 0.14	Re: 0.3888888888888889
Query: 7	Pr: 0.16	Re: 0.8888888888888888
Query: 8	Pr: 0.06	Re: 0.75
Query: 9	Pr: 0.12	Re: 0.75
Query: 10	Pr: 0.08	Re: 0.16666666666666666


Avg precision:0.1
Avg Recall:0.43
Top 100 documents in rank list
Query: 1	Pr: 0.0	Re: 0.0
Query: 2	Pr: 0.09	Re: 0.6
Query: 3	Pr: 0.09	Re: 0.6
Query: 4	Pr: 0.06	Re: 0.3333333333333333
Query: 5	Pr: 0.12	Re: 0.631578947368421
Query: 6	Pr: 0.08	Re: 0.4444444444444444
Query: 7	Pr: 0.09	Re: 1.0
Query: 8	Pr: 0.03	Re: 0.75
Query: 9	Pr: 0.06	Re: 0.75
Query: 10	Pr: 0.04	Re: 0.16666666666666666
Avg precision:0.07
Avg Recall:0.53

Top 500 documents in rank list
Query: 1	Pr: 0.002	Re: 1.0
Query: 2	Pr: 0.03	Re: 1.0
Query: 3	Pr: 0.03	Re: 1.0
Query: 4	Pr: 0.032	Re: 0.8888888888888888
Query: 5	Pr: 0.038	Re: 1.0
Query: 6	Pr: 0.034	Re: 0.9444444444444444
Query: 7	Pr: 0.018	Re: 1.0
Query: 8	Pr: 0.008	Re: 1.0
Query: 9	Pr: 0.016	Re: 1.0
Query: 10	Pr: 0.026	Re: 0.5416666666666666
Avg precision:0.02
Avg Recall:0.94