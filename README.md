# AI Resume Screening and Scoring System 

The AI Resume Screening and Scoring System is a machine learning-based application designed to analyze resumes, extract relevant skills, evaluate resume quality, and compute job matching scores. The system helps automate the initial stage of recruitment by reducing manual effort and improving screening accuracy.

## Features
- Resume upload and analysis
- Automatic skill extraction
- Resume quality scoring
- Job and resume similarity matching
- Machine learning model integration
- Web-based interface
- Backend implemented using Python and Flask

## Technologies Used
- Python
- Flask
- Machine Learning
- Natural Language Processing
- scikit-learn
- HTML
- Pickle

## Project Structure
The repository is named `ai_generative` and contains the following layout:

```
ai_generative/
│
├── app.py
├── requirements.txt
├── resume_model.pkl  
├── procdata.csv  
├── static/          
├── uploads/           
└── README.md
```

## How the System Works
1. The user uploads a resume (PDF).
2. Resume text is extracted.
3. Skills and keywords are identified.
4. Resume quality score is calculated.
5. The machine learning model evaluates similarity and predicts role/fit.
6. Results are displayed on the web interface.

## Machine Learning Details
- Model: Logistic regression (or other classification model) saved with `joblib`/Pickle
- Feature extraction: TF-IDF or other NLP preprocessing
- Trained model stored as `resume_model.pkl`

## Future Enhancements
- Better PDF parsing and multilingual support
- Multiple job role matching and ranking
- Skill gap analysis and personalized suggestions
- Improved UI and accessibility
- Database integration and persistent storage

---
