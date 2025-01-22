import os
import re
import glob
import nltk
import shutil
import base64
import docx2txt
import requests
import wikipedia
import pdfplumber
import numpy as np
import pandas as pd
#import wikipediaapi
import streamlit as st
from docx import Document
from pipelines import pipeline
from preprocessing import Preprocessing
from elasticsearch_QnA import Elasticsearch_qna
from langchain.docstore.document import Document
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer 

nltk.download('punkt')

# pandas display options
pd.set_option('display.max_colwidth', None)

wiki = wikipedia.set_lang('en')
#wiki = wikipediaapi.Wikipedia('en')

@st.cache_data() #allow_output_mutation=True
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('background.jpeg')

### Function to save the uploaded files:
def save_uploaded_file(uploadedfile):
	try:
		shutil.rmtree("./tempDir")
	except Exception:
		pass
	try:
		os.makedirs("./tempDir")
		os.makedirs("./tempDir/csv_files")
	except Exception:
		pass
	with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
		f.write(uploadedfile.getbuffer())
		#st.balloons()
	return st.success("Saved file : {} in tempDir folder".format(uploadedfile.name))

### Function to save the Wiki files:
def save_wiki_file(wikifile,page):
	try:
		shutil.rmtree("./tempDir")
	except Exception:
		pass
	try:
		os.makedirs("./tempDir")
		os.makedirs("./tempDir/csv_files")
	except Exception:
		pass
	
	with open(wikifile, 'w') as file:
		file.write(page)
		st.write('Page content saved as {}'.format(wikifile))

def make_clickable(ques, ans):
    return '<a href="javascript:;""{}">{}</a>'.format(ans,ques)

def get_answer(df,question,qna_model):

	corpus = list(df["question"].values)
	vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform(corpus)
	q = vectorizer.transform(question)

	for index, row in df.iterrows():
	    if(cosine_similarity(X[index],q)):
	        df["Cosine_similarity"][index] = cosine_similarity(X[index],q)
	
	if df['Cosine_similarity'].max() >= 0.30:        
		ans_index = df[df["Cosine_similarity"] == df['Cosine_similarity'].max()].index.values
		answer = df['answer'][ans_index]
		answer = list(answer)[0]
		return answer

	elif (df['Cosine_similarity'].max() >= 0.05) or (df['Cosine_similarity'].max() <= 0.30):
		result = Elasticsearch_qna(qna_model,question)
		answer = result.answering()
		return answer
	 
	else:
	    return "There is no matching answer"

def get_answer_es(question,qna_model):

	result = Elasticsearch_qna(qna_model,question)
	answer = result.answering()
	return answer

def main():
	#st.title("Dquery bot")
	title = '<p style="background-color:rgb(200,100,100);color:rgb(255,255,255);text-align:center;font-size:30px;padding:10px 10px;font-weight:bold"> 🔥 Document based Q&A chatbot 🔥 <p/>'
	st.markdown(title, unsafe_allow_html=True)
	menu = ["Home","Dquery","About"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Home")

		st.write('Hello, *World!* :sunglasses:')
		
		st.subheader("Wiki Search")
		search_word = [st.text_input("Enter your Search word here...")]
		try:
			results = wikipedia.search(search_word)
			st.write("search results for given Keyword:")
			on_click = False
			
			if st.button(results[1]):
			    on_click = results[1]
			elif st.button(results[2]):
			    on_click = results[2]
			elif st.button(results[3]):
			    on_click = results[3]

			def get_wiki_data(title, first_paragraph_only):
			    url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&titles={title}"
			    if first_paragraph_only:
				    url += "&exintro=1"
				    data = requests.get(url).json()
			    return Document(
				    page_content=list(data["query"]["pages"].values())[0]["extract"],
			        metadata={"source": f"https://en.wikipedia.org/wiki/{title}"},
		        )
			
			pattern0 = r"\n"
			pattern1 = r"Document\(page_content='|', lookup_str='',|\n\n\n== References =='"
			pattern2 = r", lookup_index=0\)|metadata=.*?\),\s*"
			pattern = r"^\[|\]$"
			pattern3 = r"(\n)+"


			from pydantic import BaseModel

			class Document(BaseModel):
				page_content: str
				metadata: dict = {}

			if on_click:
				page = [get_wiki_data(str(on_click), True)]
				page = re.sub(pattern, " ", str(page))
				page = re.sub(pattern1, " ", str(page))
				page = re.sub(pattern2, " ", str(page))
				page = re.sub(pattern3, " ", str(page)) 
				page = re.sub(pattern0, " ", str(page))

			st.success(page)
			doc_name = 'tempDir/{}.txt'.format(on_click)
			
			save_wiki_file(doc_name,page)

			# with open(doc_name, 'w') as file:
			# 	file.write(page)
			# 	st.write('Page content saved as {}.docx'.format(on_click))

			# Save the page content as a readable Word document
			# page.save('{}.docx'.format(on_click))
			# st.write('Page content saved as {}.docx'.format(on_click))
		
		except:
			pass
			#st.success(f"Page is not found for given word: {search_word}")

		st.subheader("DocumentFiles")
		docx_file = st.file_uploader("Upload File",type=['txt','docx','pdf'])
		if st.button("Process"):
			if docx_file is not None:
				file_details = {"Filename":docx_file.name,"FileType":docx_file.type,"FileSize":docx_file.size}
				st.write(file_details)
				# Check File Type
				if docx_file.type == "text/plain":
					st.text(str(docx_file.read(),"utf-8")) # empty
					raw_text = str(docx_file.read(),"utf-8") # works with st.text and st.write,used for futher processing
					# st.text(raw_text) # Works
					st.write(raw_text) # works
					save_uploaded_file(docx_file)

				elif docx_file.type == "application/pdf":
					try:
						with pdfplumber.open(docx_file) as pdf:
							page = pdf.pages[0]
							st.write(page.extract_text())
						save_uploaded_file(docx_file)
					except:
						st.write("None")

				elif docx_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
				# Use the right file processor ( Docx,Docx2Text,etc)
					raw_text = docx2txt.process(docx_file) # Parse in the uploadFile Class directory
					st.write(raw_text)
					save_uploaded_file(docx_file)

	elif choice == "Dquery":

		st.subheader("Dquery bot")
		st.write('Hello, *World!* :sunglasses:')
		st.set_option('deprecation.showfileUploaderEncoding', False)
		@st.cache_resource        #cache(allow_output_mutation=True)
		def load_model(model):
			if model == "Auto_QnA":
				model = pipeline("question-generation") #question-generation
			elif model == "QnA":
				model = pipeline("multitask-qa-qg") ## model="valhalla/t5-base-qa-qg-hl"
			else:
				model = "Model failed to load"
			return model
		if st.button("Start"):
			try:
				path = os.path.join(os.path.dirname(os.path.realpath("tempDir")),"tempDir")
				files = next(os.walk(path))[2]
				file = ''.join(files)
				file_path = os.path.join(path,file)

			except UnboundLocalError:
				st.error("Session Restarted, please provide file in home page!!!")
				st.stop()
			except FileNotFoundError:
				st.error("Please provide file in home page!!!")
				st.stop()
			with st.spinner("Loading model into memory..."):
				model = "Auto_QnA"
				model = load_model(model)
				st.success("Models loaded into memory successfully!!!")
				st.balloons()
				
			try:
				st.write(model)
				obj = Preprocessing(model,file_path)
				df = obj.main()
				if isinstance(df,pd.DataFrame):
					df = df.to_csv(index=False)
				csv_path = os.path.join(os.path.dirname(os.path.realpath("tempDir/csv_files")),"csv_files")
				file_name = file.split('.')[0]
				csv_file_name = file_name +"_auto_QnA.csv"
				with open(os.path.join(csv_path,csv_file_name),"w") as f:
					f.write(df+ "\n")
				st.success("Model got trained on given document successfully!!!")
			except UnboundLocalError:
				st.error("Session Restarted, please provide the file in home page!!!")
				st.stop()

		csv_path = os.path.join(os.path.dirname(os.path.realpath("tempDir/csv_files")),"csv_files")
		if any(File.endswith(".csv") for File in os.listdir(csv_path)):
			#try:
			csv_files = next(os.walk(csv_path))[2]
			csv_file = ''.join(csv_files)
			csv_file_path = os.path.join(csv_path,csv_file)
			df_qna = pd.read_csv(csv_file_path)
			df_qna.reset_index(inplace=True)
			df_qna['question'] = (df_qna.index + 1).astype(str) + '. ' + df_qna['question']
			df_qna["Cosine_similarity"] = 0 
			question = [st.text_input('Enter your questions here...')]

			try:
				if question != ['']:
					st.write("Response :")
					with st.spinner("Searching for answers..."):
						question = list(question)
						model = "QnA"
						model = load_model(model)
						answer = get_answer(df_qna,question,model)
						st.info(answer)

				# --- Initialising SessionState ---
				if "load_state" not in st.session_state:
					st.session_state.load_state = False

				if st.button('Generated Questions') or st.session_state.load_state:
					st.session_state.load_state = True
					for index, row in df_qna.iterrows():
						question = row['question']
						answer = row['answer']
						st.write(question)
						if st.checkbox(question):
							st.success(f"Answer : {answer}")

					# 	if st.button(question, key=f'link-{question}'):
					# 		st.session_state.clicked_link = question
					# 		st.success(f"{answer}")
					
				# df_qna['question'] = df_qna.apply(lambda x: make_clickable(x['question'], x['answer']), axis=1)
				# question_df = pd.DataFrame(df_qna['question'])
				# question_df.reset_index(inplace = True)
				# question_df.set_index('index', inplace=True)
				# question_html = question_df.to_html(escape=False)
				# st.write(question_html, unsafe_allow_html=True)

			except pd.errors.EmptyDataError:
				#st.error("Auto questions generation failed!!")
				#st.stop()
				question = [st.text_input('Enter your questions here...')]
				if question != ['']:
					st.write("Response :")
					with st.spinner("Searching for answers..."):
						question = list(question)
						model = "QnA"
						model = load_model(model)
						answer = get_answer_es(question,model)
						st.info(answer)
					
	else:
		st.subheader("About")
		st.text("Dquery bot for Document based Question and Answering")

if __name__ == '__main__':
	main()
