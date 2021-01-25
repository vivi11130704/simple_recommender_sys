Author: Wei Wang

Date: 1/24/2020

# Project Goal

The purpose of this project is to build and deploy a simple recommender algorithm that will recommend the top 20 most relevant products to a customer.

# Code Usage
1. git clone this repo to a local desktop

2. `cd` into this repo. Build a docker image by running this command: `docker build -t myimagename ./`

3. create a docker container from the image by running this command: `docker run -p 8080:8080 -d myimagename`

4. a recommendation list can be fetched by using commands like: `curl -d '{"user_id_hashes":["b9cbac77a336d62efd54404d2bccaecd"]}' -H "Content-Type: application/json" -X POST http://0.0.0.0:8080/invocations`

Note that the program would fail if use `http://localhost:8080`. Need to use `http://0.0.0.0:8080` instead.

# Model Description

This project builds a content based filtering recommendation algorithm using product text attributes. Specifically:

1. Combine, vectorize and TF-IDF transform text information of products to form a product profile for each item.

2. Construct a user profile for each user based on the items and the quantities that the user has purchased. 

3. Calculate the cosine similarity between the user profile and each products. Select the top 20 items that the user hasn't purchased yet to form a recommendation list.

4. For new users who do not have user profiles established, recommend 20 most popular items.

The model is trained on a local desktop and pickled to be used in a production environment. 

# File References

**Model Development:**
"/development/Recommender System.ipynb"

This notebook illustrates a step-by-step walk-through of the model development process. It also discusses the limitations and the future possible developments.

**Data Folder:**
"/src/data"

"recommendations_take_home.csv" is the source data. 

"products.csv" is a dataset generates from the "recommendations_take_home.csv" which contains non-duplicate product information.

**Saved Model Folder:**
"/src/model"

"item_ids.pickle", "tfidf_feature.pickle" and "tfidf_matrix.pickle" are product profiles that are created by the "Recommender System.ipynb" notebook.

"most_popular_list.pickle" includes the top 20 most popular items for new users.

**Model Code:**
"/src/utils/utils.py"

The code in "utils.py" is very similar as the "Recommender System.ipynb" except that it has been adjusted for better model deployment.

**API Creation:**
"run.py"

Create a flask API to consume inputs and generates model outputs.

**Docker Deployment**
"Dockerfile" and "requirements.txt"
