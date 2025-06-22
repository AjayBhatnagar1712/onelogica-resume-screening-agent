# 🚀 Onelogica Resume Screening Agent

## 📌 Project Overview

This project is a submission for the Onelogica AI Internship Challenge, Use Case 1: "Autonomous Resume Screening Agent for HR Teams." It demonstrates an agentic approach that reads resumes, extracts relevant information, matches them against a given Job Description (JD), ranks the top candidates, and explains the reasoning.

## 🎯 Objective

To autonomously:

* Accept a JD as input
* Parse and embed resumes from a directory
* Rank resumes based on semantic similarity
* Provide human-readable summaries and match scores

## 🧠 Technologies & Tools Used

* **Python**
* **Streamlit** – Web frontend
* **LangChain** – LLM integration and orchestration
* **ChromaDB** – Lightweight Vector Database for fast similarity search
* **OpenAI API (GPT-3.5/4)** – Resume and JD embedding, summarization
* **PyPDF2 / LangChain PDF Loader** – PDF extraction

## ⚙️ How It Works

1. **Job Description Input**: User pastes a JD into the app.
2. **Resume Parsing**: Resumes in `/resumes` folder are read and converted to text.
3. **Embedding**: JD and resumes are embedded using OpenAI Embedding model.
4. **Similarity Matching**: Cosine similarity is calculated between JD and each resume.
5. **Top 3 Selection**: Resumes are ranked by match score.
6. **Summary Explanation**: Each selected resume is explained using summarization.

## 📂 Folder Structure

```
Onelogica_Resume_Screening_Agent/
├── app.py
├── main.py
├── requirements.txt
├── resumes/
├── architecture_diagram.png
├── explanation.pdf
├── sample_outputs/
├── README.md
```

## ✅ Steps to Run

```bash
# 1. Clone the repo
$ git clone https://github.com/yourusername/onelogica-resume-agent
$ cd onelogica-resume-agent

# 2. Create virtual environment
$ python -m venv venv
$ source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
$ pip install -r requirements.txt

# 4. Run the app
$ streamlit run app.py
```

## 📄 Sample Output

* Top 3 ranked resumes with match scores
* Natural language explanation (summary) of why they were selected

## 🧠 Agentic Behavior

* Autonomous loop to embed, search, rank, and explain resumes
* Minimal human input required beyond JD entry

## 📬 Contact

Ajay Bhatnagar
ajaybhatnagar1712@gmail.com
