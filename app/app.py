from flask import Flask, render_template, request
import torch
import pickle
import numpy as np

# Define Skipgram model class
class SkipgramModel(torch.nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(SkipgramModel, self).__init__()
        self.embedding_v = torch.nn.Embedding(vocab_size, emb_size)
        self.embedding_u = torch.nn.Embedding(vocab_size, emb_size)

    def forward(self, center_words, target_words, all_vocabs):
        center_embeds = self.embedding_v(center_words)
        target_embeds = self.embedding_u(target_words)
        all_embeds = self.embedding_u(all_vocabs)
        
        scores = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        norm_scores = all_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        
        # Negative log likelihood
        nll = -torch.mean(torch.log(torch.exp(scores) / torch.sum(torch.exp(norm_scores), 1).unsqueeze(1)))
        return nll

# Initialize Flask app
app = Flask(__name__)

# Load word2index
with open('word2index.pkl', 'rb') as f:
    word2index = pickle.load(f)

# Check the vocab size to make sure it's 73250 (including <UNK>)
vocab_size = len(word2index)
print(f"Vocabulary size in app.py: {vocab_size}")  # This should print 73250

# Use vocab_size = 73249 for model initialization (to match the trained model)
skipgram_model = SkipgramModel(vocab_size - 1, 2)  # Subtract 1 for <UNK>

# Load the saved model
skipgram_model.load_state_dict(torch.load('skipgram_model.pth'))
skipgram_model.eval()  # Set model to evaluation mode

# Helper function to prepare the input query for the model
def prepare_sequence(seq, word2index):
    return torch.LongTensor([word2index.get(word, word2index['<UNK>']) for word in seq])

# Function to compute similarity between query and words in the corpus
def compute_similarity(query, top_n=10):
    # Pre-process the query
    query_words = query.lower().split()
    
    # Prepare sequence with word indices, handling out-of-vocabulary words
    query_indices = [word2index.get(word, word2index.get("<UNK>")) for word in query_words]

    # Get the embeddings for the query words
    query_embedding = skipgram_model.embedding_v(torch.LongTensor(query_indices))

    # Normalize query embedding
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)

    similarities = []
    for idx, word in enumerate(word2index):
        # Ensure that the index is within the valid range for the embeddings
        if idx < vocab_size - 1:  # Exclude <UNK>
            word_embedding = skipgram_model.embedding_u(torch.LongTensor([idx]))
            word_embedding = torch.nn.functional.normalize(word_embedding, p=2, dim=1)
            
            similarity = torch.nn.functional.cosine_similarity(query_embedding, word_embedding)
            
            # Limit the similarity between 0 and 1 to avoid values greater than 1.0
            similarity = torch.clamp(similarity, min=0.0, max=1.0)

            similarities.append((word, similarity.item()))

    # Sort by similarity and return only the top N most similar words
    similarities.sort(key=lambda x: x[1], reverse=True)
    # Return only the words
    return [word for word, _ in similarities[:top_n]]


# Flask route to handle search query input
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        if query.strip() == "":  # Check if query is empty
            return render_template("index.html", query=None, top_similar=None, error_message="Please enter a word to search.")
        top_similar = compute_similarity(query)
        return render_template("index.html", query=query, top_similar=top_similar)
    return render_template("index.html", query=None, top_similar=None)


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
