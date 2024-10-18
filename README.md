# Biomedical Chatbot Website
This project is a Flask-based chatbot website that assists biomedical engineers by answering questions from medical device manuals. The website integrates user authentication, PDF processing, embedding techniques (FAISS), and a question-answering model.

## Features

- User registration and authentication
- Chatbot that answers questions based on medical device manuals (PDF)
- Navigation for educational resources on biomedical devices
- Integration with Hugging Face models for embeddings and language understanding

## Project Architecture

- **Frontend**: HTML templates rendered using Flask's `render_template()`
- **Backend**: Flask web framework with SQLite/MySQL (using SQLAlchemy ORM) and Flask-Login for user management.
- **PDF Embedding & Retrieval**: LangChain, Hugging Face Embeddings, FAISS (vector search engine)
- **Question Answering**: Hugging Face Model (`flan-t5-large`)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/biomedical-chatbot.git
cd biomedical-chatbot
```
### 2. Set Up Python Virtual Environment
Install Python (version 3.8 or above is recommended). Then, create a virtual environment:

```bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install the Dependencies
Make sure to install the required packages using pip:

```bash
Copy code
pip install -r requirements.txt
```
### 4. Set Up Environment Variables
Create a .env file at the root of the project and configure the following environment variables:

```makefile
Copy code
FLASK_APP=run.py
SECRET_KEY=<your_secret_key>
DATABASE_URL=sqlite:///site.db  # For SQLite; you can configure for MySQL/PostgreSQL if needed.
HUGGINGFACE_API_TOKEN=<your_huggingface_api_token>
```
### 5. Initialize the Database
Run the following commands to create and initialize the database:

```bash
Copy code
flask db init
flask db migrate
flask db upgrade
```

### 6. Download and Prepare PDF Files
Place the PDF files of medical device manuals under the app/pdfs directory. Modify the path in the PyPDFLoader section in the code to reference the correct PDF file.

### 7. Run the Flask Application
Start the Flask development server:

``` bash
Copy code
flask run
The website should now be running locally at http://127.0.0.1:5000/.
```

## Usage Guidelines
Register/Login: Create an account or log in using your credentials.
Chatbot: Navigate to the chat section and input questions related to biomedical devices. The chatbot will answer based on the embedded PDFs.
Educational Resources: Explore the "Educational" section that provides overviews and explanations of various biomedical devices in the categories of Imaging, ICU, Laboratory, and Surgical.
Project Structure
```plaintext
Copy code
biomedical-chatbot/
│
├── app/
│   ├── __init__.py           # Flask app factory
│   ├── models.py             # Database models (User)
│   ├── forms.py              # Registration, Login Forms
│   ├── routes.py             # Defines all routes for the app
│   ├── templates/            # HTML files for rendering views
│   ├── static/               # Static assets like CSS, JS, images
│   ├── pdfs/                 # PDF files for embedding (medical manuals)
│   ├── langchain.py          # LangChain integration for PDF extraction
│   └── ...                   # Other auxiliary files
│
├── migrations/               # Database migrations
├── venv/                     # Virtual environment files
├── README.md                 # This file
├── requirements.txt          # Python package dependencies
└── run.py                    # Entry point for running the Flask app

```
## API Endpoints
/home: Homepage
/chat: Chatbot interface for querying medical device manuals
/educational: Educational resources on biomedical devices
/register: User registration page
/login: User login page
/logout: Logs out the current user
/get_response: API endpoint for handling chatbot responses
