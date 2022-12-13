"""
NLP Project to Build a Resume Parser in Python using Spacy

his script uses the spacy library to parse the text in the file resume.txt and extract the entities and their labels. The entities are then printed to the console. You could further extend this script to extract specific information from the resume, such as the person's name, email address, and work experience.
"""
import spacy

# load the spacy model
nlp = spacy.load('en_core_web_sm')

# read in the resume text
with open('resume.txt', 'r') as f:
    resume_text = f.read()

# parse the resume text using spacy
resume_doc = nlp(resume_text)

# extract the entities and their labels
entities = [(e.text, e.label_) for e in resume_doc.ents]

# print the entities
print(entities)


