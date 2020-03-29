# Cranfield Corpus Vector Space Model

This project was done as a homework assignment for CS582: Information Retrieval course at the University of Illinois at Chicago during the Spring 2020 term.

----

The [dataset](cranfieldDocs "Cranfield collection") used was the Cranfield collection which is a standard Information Retrieval text collection, consisting of 1400 documents from the aerodynamics field.

Each document in the collection are in SGML format. So the SGML tags (e.g.,\<TITLE>, \<DOC>,\<TEXT>, etc.) have been eliminated and only the text between the  \<TITLE> and \<TEXT> tags have been retained for building the Vector Space Retrieval Model.

For the tasks in this assignment the pre-processing tools implemented in [this project](https://github.com/samujjwaal/CiteSeer-Text-Processing "CiteSeer Collection Text Processing") were reused. The documents after cleaning were stored in a new [folder](preprocessed_cranfieldDocs).

There is also a [list of queries](queries.txt) and a [list of relevant documents](relevance.txt) for each query provided, to calculate the precision and recall values of the retrieval model. The evaluation metrics are saved in an [output](output.txt) file.

The same text pre-processing operations were applied on both the corpus documents and the queries.

---

The tasks in the [assignment](Tasks.pdf "Assignment description") included:

1. Implement an indexing scheme based on the vector space model. *TF-IDF* used for weighting scheme.

2. For each of the queries, determine a ranked list of documents, in descending order of their *cosine similarity* with the queries.

3. Determine the average precision and recall for the queries, using top 10, 50, 100 and 500 documents in the ranking.


Check out the [Jupyter Notebook](cranfield.ipynb "Cranfield Corpus Vector Space Model") to see the python implementation of the tasks.
