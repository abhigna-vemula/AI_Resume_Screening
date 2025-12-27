from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import PyPDF2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import traceback
import re
from collections import Counter

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
try:
    model = joblib.load("resume_model.pkl")
    tfidf = model.named_steps['tfidf']
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tfidf = None

# Common skills database
SKILLS_DATABASE = {
    'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust', 'typescript'],
    'web': ['html', 'css', 'react', 'angular', 'vue', 'nodejs', 'express', 'django', 'flask', 'spring', 'asp.net'],
    'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'redis', 'cassandra', 'dynamodb'],
    'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible'],
    'data_science': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'nlp', 'computer vision'],
    'tools': ['git', 'jira', 'agile', 'scrum', 'ci/cd', 'linux', 'bash', 'rest api', 'graphql'],
    'soft_skills': ['leadership', 'communication', 'teamwork', 'problem solving', 'analytical', 'project management']
}

# Role categories
ROLE_CATEGORIES = {
    'Software Developer': ['python', 'java', 'javascript', 'git', 'agile', 'rest api', 'sql'],
    'Data Scientist': ['python', 'machine learning', 'pandas', 'numpy', 'sql', 'tensorflow', 'deep learning'],
    'Frontend Developer': ['html', 'css', 'javascript', 'react', 'angular', 'vue', 'typescript'],
    'Backend Developer': ['python', 'java', 'nodejs', 'sql', 'mongodb', 'rest api', 'docker'],
    'DevOps Engineer': ['aws', 'docker', 'kubernetes', 'jenkins', 'terraform', 'linux', 'ci/cd'],
    'Full Stack Developer': ['javascript', 'react', 'nodejs', 'mongodb', 'html', 'css', 'python'],
    'ML Engineer': ['python', 'tensorflow', 'pytorch', 'machine learning', 'deep learning', 'docker', 'aws'],
    'Mobile Developer': ['swift', 'kotlin', 'react', 'java', 'ios', 'android'],
}

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""

def extract_skills(text):
    """Extract skills from resume text"""
    text_lower = text.lower()
    found_skills = {}
    
    for category, skills in SKILLS_DATABASE.items():
        category_skills = []
        for skill in skills:
            if skill.lower() in text_lower:
                category_skills.append(skill)
        if category_skills:
            found_skills[category] = category_skills
    
    return found_skills

def calculate_resume_quality_score(resume_text):
    """Calculate standalone resume quality score (0-100)"""
    score = 0
    max_score = 100
    feedback = []
    
    text_lower = resume_text.lower()
    word_count = len(resume_text.split())
    
    # 1. Length Check (15 points)
    if word_count >= 300:
        score += 15
        feedback.append({"category": "Length", "score": 15, "max": 15, "status": "excellent"})
    elif word_count >= 200:
        score += 10
        feedback.append({"category": "Length", "score": 10, "max": 15, "status": "good"})
    elif word_count >= 100:
        score += 5
        feedback.append({"category": "Length", "score": 5, "max": 15, "status": "needs_improvement"})
    else:
        feedback.append({"category": "Length", "score": 0, "max": 15, "status": "poor"})
    
    # 2. Contact Information (10 points)
    contact_score = 0
    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text):
        contact_score += 5
    if re.search(r'\b\d{10}\b|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', resume_text):
        contact_score += 5
    score += contact_score
    feedback.append({"category": "Contact Info", "score": contact_score, "max": 10, "status": "excellent" if contact_score == 10 else "needs_improvement"})
    
    # 3. Section Structure (20 points)
    sections = ['experience', 'education', 'skills', 'projects']
    section_score = 0
    found_sections = []
    for section in sections:
        if section in text_lower:
            section_score += 5
            found_sections.append(section)
    score += section_score
    feedback.append({"category": "Sections", "score": section_score, "max": 20, "status": "excellent" if section_score >= 15 else "good" if section_score >= 10 else "needs_improvement"})
    
    # 4. Skills Diversity (20 points)
    skills = extract_skills(resume_text)
    skill_categories = len(skills)
    total_skills = sum(len(s) for s in skills.values())
    
    skill_score = 0
    if total_skills >= 10:
        skill_score += 10
    elif total_skills >= 5:
        skill_score += 7
    elif total_skills >= 3:
        skill_score += 4
    
    if skill_categories >= 4:
        skill_score += 10
    elif skill_categories >= 2:
        skill_score += 6
    elif skill_categories >= 1:
        skill_score += 3
    
    score += skill_score
    feedback.append({"category": "Skills", "score": skill_score, "max": 20, "status": "excellent" if skill_score >= 15 else "good" if skill_score >= 10 else "needs_improvement"})
    
    # 5. Quantifiable Achievements (15 points)
    metrics = re.findall(r'\d+%|\d+\+|\$\d+|\d+x|\d+\s*(?:years|months|projects|users|clients)', text_lower)
    metric_score = 0
    if len(metrics) >= 5:
        metric_score = 15
    elif len(metrics) >= 3:
        metric_score = 10
    elif len(metrics) >= 1:
        metric_score = 5
    
    score += metric_score
    feedback.append({"category": "Quantifiable Results", "score": metric_score, "max": 15, "status": "excellent" if metric_score >= 10 else "good" if metric_score >= 5 else "needs_improvement"})
    
    # 6. Action Verbs (10 points)
    action_verbs = ['developed', 'created', 'designed', 'implemented', 'managed', 'led', 'improved', 
                    'increased', 'reduced', 'optimized', 'built', 'launched', 'achieved', 'delivered']
    verb_count = sum(1 for verb in action_verbs if verb in text_lower)
    
    verb_score = 0
    if verb_count >= 8:
        verb_score = 10
    elif verb_count >= 5:
        verb_score = 7
    elif verb_count >= 3:
        verb_score = 4
    
    score += verb_score
    feedback.append({"category": "Action Verbs", "score": verb_score, "max": 10, "status": "excellent" if verb_score >= 8 else "good" if verb_score >= 5 else "needs_improvement"})
    
    # 7. Professional Formatting (10 points)
    format_score = 10  # Assume good formatting if PDF is well-structured
    if word_count < 100:
        format_score = 5
    score += format_score
    feedback.append({"category": "Format", "score": format_score, "max": 10, "status": "excellent" if format_score >= 8 else "good"})
    
    # Overall rating
    if score >= 80:
        rating = "Excellent"
        rating_color = "success"
    elif score >= 60:
        rating = "Good"
        rating_color = "warning"
    elif score >= 40:
        rating = "Fair"
        rating_color = "info"
    else:
        rating = "Needs Improvement"
        rating_color = "danger"
    
    return {
        'score': round(score, 2),
        'maxScore': max_score,
        'rating': rating,
        'ratingColor': rating_color,
        'feedback': feedback,
        'wordCount': word_count,
        'totalSkills': total_skills,
        'skillCategories': skill_categories
    }

def calculate_role_match(skills_text):
    """Calculate role matching percentages"""
    text_lower = skills_text.lower()
    role_matches = {}
    
    for role, required_skills in ROLE_CATEGORIES.items():
        matched_skills = sum(1 for skill in required_skills if skill.lower() in text_lower)
        match_percentage = round((matched_skills / len(required_skills)) * 100, 2)
        role_matches[role] = {
            'percentage': match_percentage,
            'matched': matched_skills,
            'total': len(required_skills)
        }
    
    # Sort by percentage
    role_matches = dict(sorted(role_matches.items(), key=lambda x: x[1]['percentage'], reverse=True))
    return role_matches

def generate_ai_suggestions(resume_text, job_description, match_score):
    """Generate AI-powered suggestions for resume improvement"""
    suggestions = []
    resume_lower = resume_text.lower()
    job_lower = job_description.lower()
    
    # Extract important keywords from job description
    job_keywords = set(re.findall(r'\b[a-z]{3,}\b', job_lower))
    resume_keywords = set(re.findall(r'\b[a-z]{3,}\b', resume_lower))
    
    missing_keywords = job_keywords - resume_keywords
    
    if match_score < 50:
        suggestions.append({
            'type': 'critical',
            'title': 'Low Match Score',
            'message': 'Your resume has significant gaps. Consider highlighting relevant experience and skills that match the job description.'
        })
    elif match_score < 70:
        suggestions.append({
            'type': 'warning',
            'title': 'Moderate Match',
            'message': 'Good foundation, but there\'s room for improvement. Focus on emphasizing relevant skills and experiences.'
        })
    else:
        suggestions.append({
            'type': 'success',
            'title': 'Strong Match',
            'message': 'Excellent alignment with job requirements! Your resume demonstrates relevant qualifications.'
        })
    
    # Skill-based suggestions
    if 'experience' not in resume_lower and 'work' not in resume_lower:
        suggestions.append({
            'type': 'warning',
            'title': 'Missing Experience Section',
            'message': 'Add a detailed work experience section highlighting your achievements and responsibilities.'
        })
    
    if 'project' not in resume_lower:
        suggestions.append({
            'type': 'info',
            'title': 'Add Projects',
            'message': 'Include relevant projects to showcase your practical skills and initiative.'
        })
    
    # Missing keywords
    if missing_keywords and len(missing_keywords) > 5:
        important_missing = list(missing_keywords)[:5]
        suggestions.append({
            'type': 'info',
            'title': 'Keywords to Consider',
            'message': f'Consider incorporating these relevant terms: {", ".join(important_missing)}'
        })
    
    # Quantification check
    if not re.search(r'\d+%|\d+\s*(years|months|projects)', resume_lower):
        suggestions.append({
            'type': 'warning',
            'title': 'Add Quantifiable Achievements',
            'message': 'Include numbers and metrics to quantify your achievements (e.g., "Improved performance by 30%").'
        })
    
    return suggestions

def predict_selection(match_score, role_matches):
    """Predict likelihood of selection"""
    top_role_match = max(role_matches.values(), key=lambda x: x['percentage'])['percentage']
    
    # Combined scoring
    combined_score = (match_score * 0.6) + (top_role_match * 0.4)
    
    if combined_score >= 75:
        prediction = {
            'status': 'High Chance',
            'probability': round(combined_score, 2),
            'message': 'Strong candidate! Your profile aligns well with the requirements.',
            'color': 'success'
        }
    elif combined_score >= 55:
        prediction = {
            'status': 'Moderate Chance',
            'probability': round(combined_score, 2),
            'message': 'Good potential. Emphasize relevant skills in your application.',
            'color': 'warning'
        }
    else:
        prediction = {
            'status': 'Low Chance',
            'probability': round(combined_score, 2),
            'message': 'Consider gaining more relevant experience or skills for this role.',
            'color': 'danger'
        }
    
    return prediction

@app.route('/')
def index():
    """Serve the main HTML page"""
    return app.send_static_file('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    """Comprehensive resume analysis"""
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume uploaded'}), 400
        
        file = request.files['resume']
        job_description = request.form.get('jobDescription', '')
        company_name = request.form.get('companyName', 'this company')
        
        if not job_description:
            return jsonify({'error': 'Job description is required'}), 400
        
        if model is None or tfidf is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Save and process file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Extract text
        resume_text = extract_text_from_pdf(filepath)
        
        if not resume_text.strip():
            os.remove(filepath)
            return jsonify({'error': 'Could not extract text from PDF'}), 400
        
        # Calculate similarity score
        try:
            resume_vector = tfidf.transform([resume_text])
            job_vector = tfidf.transform([job_description])
            similarity = cosine_similarity(resume_vector, job_vector)[0][0]
            match_score = round(similarity * 100, 2)
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            match_score = 0
        
        # Calculate standalone resume quality score
        quality_score = calculate_resume_quality_score(resume_text)
        
        # Extract skills
        skills = extract_skills(resume_text)
        
        # Calculate role matches
        role_matches = calculate_role_match(resume_text)
        
        # Generate AI suggestions
        suggestions = generate_ai_suggestions(resume_text, job_description, match_score)
        
        # Predict selection
        selection_prediction = predict_selection(match_score, role_matches)
        
        # Prepare response
        response = {
            'filename': file.filename,
            'matchScore': match_score,
            'qualityScore': quality_score,
            'skills': skills,
            'roleMatches': role_matches,
            'suggestions': suggestions,
            'selectionPrediction': selection_prediction,
            'companyName': company_name
        }
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in analyze_resume: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/screen-multiple', methods=['POST'])
def screen_multiple():
    """Screen multiple resumes"""
    try:
        if 'resumes' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('resumes')
        job_description = request.form.get('jobDescription', '')
        
        if not job_description:
            return jsonify({'error': 'Job description is required'}), 400
        
        if model is None or tfidf is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        results = []
        
        for file in files:
            if file.filename.endswith('.pdf'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                
                resume_text = extract_text_from_pdf(filepath)
                
                try:
                    resume_vector = tfidf.transform([resume_text])
                    job_vector = tfidf.transform([job_description])
                    similarity = cosine_similarity(resume_vector, job_vector)[0][0]
                    match_score = round(similarity * 100, 2)
                except Exception as e:
                    match_score = 0
                
                # Calculate quality score
                quality_score = calculate_resume_quality_score(resume_text)
                
                results.append({
                    'filename': file.filename,
                    'score': match_score,
                    'qualityScore': quality_score['score'],
                    'status': 'Selected' if match_score >= 60 else 'Rejected'
                })
                
                os.remove(filepath)
        
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({'results': results})
    
    except Exception as e:
        print(f"Error in screen_multiple: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)