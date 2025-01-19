# A1 - That’s What I LIKE
# Word2Vec and GloVe Models with Flask Web App

This project demonstrates the use of Word2Vec and GloVe embeddings for natural language processing tasks such as word analogies and word similarity. A Flask web application is used to provide an interface to interact with the models.

## Files Overview

- `README.md`: This file, containing documentation on how to use the project.
- `requirements.txt`: Contains the necessary Python packages required for the project.
- `app.py`: The main Flask web application to run the model and display results.
- `A1.ipynb`: Jupyter notebook used for training the models and running various tasks (e.g., word analogies, similarity).
- `templates/index.html`: HTML file for the Flask app’s front-end.
- `glove.6B.100d.txt`: Pre-trained GloVe embeddings used in the notebook for training.
- `skipgram_model.pth`: A saved model file from the notebook for Word2Vec.
- `word2index.pkl`: A pickle file containing a mapping from words to indices used by the model.
- `word-test.v1.txt`: File used for word analogy tasks in the notebook.
- `wordsim_similarity_goldstandard.txt`: File used for word similarity tasks in the notebook.
- `Report on Word2Vec and GloVe Models.pdf`: A report that describes the tasks and models used in the project.

## Setup and Running the Application

**1. Clone the reposity:** 
Clone the repository to your local machine.
```bash
git clone <repository_url>
cd <repository_name>
```

**2. Install dependencies:** 
Install the dependencies listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

**3. Running the Flask App:**
To start the Flask web application, run the following command:
```bash
python app/app.py
```
This will start the app on `http://127.0.0.1:5000/`.

## Features of the Web Application
- **Word2Vec and GloVe Model Interaction**: The web app lets you input a word and get 10 nearest neighbors based on the trained embeddings.
  
- **Visualization**: Results from the model (e.g., similar words) will be displayed on the webpage.