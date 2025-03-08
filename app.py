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
import pytesseract

# Configurer la clé API OpenAI
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Fonction pour extraire le texte du PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
        else:
            # Si le texte n'est pas trouvé, essayer d'extraire le texte des images
            for image in page.images:
                img = Image.open(io.BytesIO(image.data))
                text += pytesseract.image_to_string(img)
    return text

# Fonction pour extraire la première page du PDF
def extract_first_page(pdf_file):
    reader = PdfReader(pdf_file)
    first_page = reader.pages[0]
    for image_file_object in first_page.images:
        return image_file_object.data
    return None

PDF_PATH = 'C:\\Users\\user\\Desktop\\pdf_dgi\\pdf_dgi.pdf'  # Chemin direct vers le PDF

def main():
    st.image('https://assets.medias24.com/images/CAN2025/JOUEURS/GENERAL/logo_medias24.jpeg', width=200)
    st.header('Explorez le rapport sur les mesures fiscales de la Loi de Finances 2025 à travers notre chatbot')
    
    # Charger le PDF directement
    try:
        with open(PDF_PATH, 'rb') as pdf_file:
            pdf_text = extract_text_from_pdf(pdf_file)
            first_page_image = extract_first_page(pdf_file)
            
            # Diviser le texte en morceaux
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(pdf_text)
            
            # Créer des embeddings
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            
            # Initialiser le chatbot
            llm = OpenAI(temperature=0)
            qa = ConversationalRetrievalChain.from_llm(llm, knowledge_base.as_retriever())
            
            # Afficher la première page du PDF
            if first_page_image:
                st.image(io.BytesIO(first_page_image))
            else:
                st.warning('Aucune image trouvée sur la première page')
            
            # Générer un résumé
            summary_prompt = "Génère un résumé concis du document en 3 à 5 phrases."
            chat_history = []
            summary = qa.run({'question': summary_prompt, 'chat_history': chat_history})
            st.subheader('Résumé')
            st.write(summary)
            
            # Générer 5 questions pertinentes
            questions_prompt = "Propose 5 questions pertinentes basées sur le contenu du document."
            questions = qa.run({'question': questions_prompt, 'chat_history': chat_history})
            
            # Display suggested questions as buttons
            questions = questions.split('\n')
            for question in questions:
                if question.strip():  # Ensure the question is not empty
                    if st.button(question):
                        response = qa.run({'question': question, 'chat_history': chat_history})
                        st.write(response)
            
            # Interface de chat
            st.write('Posez vos questions sur le PDF :')
            user_question = st.text_input('Votre question')
            if user_question:
                response = qa.run({'question': user_question, 'chat_history': chat_history})
                st.write(response)
    except FileNotFoundError:
        st.error(f'Le fichier PDF n\'a pas été trouvé à l\'emplacement : {PDF_PATH}')
        return

if __name__ == "__main__":
    main()
