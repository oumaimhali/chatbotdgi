import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from config import OPENAI_API_KEY
from PIL import Image
import io

# Configurer la clé API OpenAI
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Fonction pour extraire le texte du PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Fonction pour extraire la première page du PDF
def extract_first_page(pdf_file):
    reader = PdfReader(pdf_file)
    first_page = reader.pages[0]
    return first_page

# Interface Streamlit
st.title('Chatbot PDF avec ChatGPT')

# Télécharger le PDF
pdf_file = st.file_uploader('Téléchargez votre PDF', type='pdf')

if pdf_file is not None:
    # Extraire le texte
    text = extract_text_from_pdf(pdf_file)
    
    # Diviser le texte en morceaux
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Créer des embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    # Initialiser le chatbot
    llm = OpenAI(temperature=0)
    qa = ConversationalRetrievalChain.from_llm(llm, knowledge_base.as_retriever())
    
    # Afficher le titre du PDF
    st.subheader(f'Titre du PDF : {pdf_file.name}')
    
    # Afficher la première page du PDF
    first_page = extract_first_page(pdf_file)
    st.image(io.BytesIO(first_page.extract_text()), caption='Page de garde')
    
    # Générer un résumé
    summary_prompt = "Génère un résumé concis du document en 3 à 5 phrases."
    summary = qa.run(summary_prompt)
    st.subheader('Résumé du PDF')
    st.write(summary)
    
    # Générer 5 questions pertinentes
    questions_prompt = "Propose 5 questions pertinentes basées sur le contenu du document."
    questions = qa.run(questions_prompt)
    st.subheader('Questions suggérées')
    st.write(questions)
    
    # Interface de chat
    st.write('Posez vos questions sur le PDF :')
    user_question = st.text_input('Votre question')
    if user_question:
        response = qa.run(user_question)
        st.write(response)
