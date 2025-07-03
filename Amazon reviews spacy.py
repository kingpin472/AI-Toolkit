# 🧾 NLP with spaCy – Amazon Reviews (NER + Sentiment)

# 1. Import spaCy
import spacy

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# 2. Sample Amazon Reviews
reviews = [
    "I love my new Apple iPhone 14. It's super fast and sleek!",
    "The Samsung Galaxy S22 camera is amazing, but battery life could be better.",
    "Terrible experience with Sony headphones. Sound was distorted.",
    "Logitech mouse works like a charm. Very comfortable.",
    "The Dell XPS laptop overheats quickly. Disappointed!"
]

# 3. Named Entity Recognition (NER)
print("🔍 Named Entities:")
for review in reviews:
    doc = nlp(review)
    print(f"\nReview: {review}")
    for ent in doc.ents:
        print(f"  - {ent.text} ({ent.label_})")

# 4. Rule-based Sentiment Analysis (Simple)
print("\n🧠 Sentiment Analysis:")
positive_keywords = ["love", "amazing", "great", "good", "works", "comfortable", "super"]
negative_keywords = ["terrible", "disappointed", "bad", "distorted", "overheats"]

for review in reviews:
    review_lower = review.lower()
    pos_hits = sum(word in review_lower for word in positive_keywords)
    neg_hits = sum(word in review_lower for word in negative_keywords)

    if pos_hits > neg_hits:
        sentiment = "Positive"
    elif neg_hits > pos_hits:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    print(f"\nReview: {review}\nSentiment: {sentiment}")
