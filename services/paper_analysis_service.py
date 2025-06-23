import fitz  # PyMuPDF
import docx
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import io

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class PaperAnalysisService:
    def __init__(self):
        # Load Sentence-BERT model
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define knowledge base: potential research directions
        self.RESEARCH_DIRECTIONS = [
            "Collect a larger, more diverse dataset to improve model robustness.",
            "Automate the manual steps using AI-assisted labeling or preprocessing.",
            "Explore real-time deployment of the model in clinical or industrial settings.",
            "Include cross-validation techniques to ensure model generalizability.",
            "Investigate combining imaging data with clinical reports (multi-modal learning).",
            "Address data imbalance or model bias by using techniques like SMOTE.",
            "Improve the explainability of model predictions using SHAP or Grad-CAM.",
            "Adapt the model for deployment in low-resource environments.",
            "Conduct further ablation studies to understand each component's contribution.",
            "Incorporate transfer learning from related domains.",
            "Evaluate long-term performance of the system under real-world conditions.",
            "Integrate user feedback mechanisms to continuously improve system accuracy."
        ]
        self.direction_embeddings = self.bert_model.encode(self.RESEARCH_DIRECTIONS)

    def extract_text_from_pdf(self, file_stream):
        """Extract text from PDF file stream"""
        try:
            text = ""
            with fitz.open(stream=file_stream, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            return f"Error extracting PDF text: {str(e)}"

    def extract_text_from_docx(self, file_stream):
        """Extract text from DOCX file stream"""
        try:
            doc = docx.Document(io.BytesIO(file_stream))
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            return f"Error extracting DOCX text: {str(e)}"

    def extract_sections(self, text):
        """Extract Future Work and Limitations sections"""
        text = re.sub(r'\s+', ' ', text)
        
        # Patterns for Future Work section
        future_work_patterns = [
            r'(Future\s+Work|Future\s+Scope|Future\s+Research|Future\s+Directions)(.*?)(Conclusion|References|Bibliography|Acknowledgment|$)',
            r'(Future\s+Work)(.*?)(Limitations|Conclusion|References|Bibliography|$)'
        ]
        
        # Patterns for Limitations section
        limitations_patterns = [
            r'(Limitations|Challenges|Shortcomings|Constraints)(.*?)(Conclusion|Future\s+Work|References|Bibliography|Acknowledgment|$)',
            r'(Limitations)(.*?)(Discussion|Conclusion|References|Bibliography|$)'
        ]
        
        fw_text = ""
        for pattern in future_work_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                fw_text = match.group(2).strip()
                break
        
        lim_text = ""
        for pattern in limitations_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                lim_text = match.group(2).strip()
                break
        
        return fw_text, lim_text

    def summarize_to_bullets(self, section_text, max_points=10):
        """Convert section text to bullet points"""
        if not section_text:
            return ["Not found."]
        
        try:
            # Try using NLTK's sentence tokenizer
            try:
                sentences = sent_tokenize(section_text)
            except:
                # Fallback to basic sentence splitting if NLTK fails
                sentences = re.split(r'(?<=[.!?])\s+', section_text)
            
            clean_sentences = [s.strip() for s in sentences if len(s.split()) > 5]
            return clean_sentences[:max_points] if clean_sentences else ["No clear points found."]
        except Exception as e:
            # Last resort fallback - just split by periods
            try:
                basic_sentences = section_text.split('.')
                clean_sentences = [s.strip() + '.' for s in basic_sentences if len(s.split()) > 5]
                return clean_sentences[:max_points] if clean_sentences else ["Unable to extract clear points."]
            except:
                return ["Error summarizing text. Please check the text format."]

    def suggest_research_directions(self, fw_text, lim_text, top_k=5):
        """Suggest research directions based on text analysis"""
        combined_text = fw_text + " " + lim_text
        if not combined_text.strip():
            return ["Not enough text to analyze."]
        
        try:
            input_embedding = self.bert_model.encode([combined_text])
            similarities = cosine_similarity(input_embedding, self.direction_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [self.RESEARCH_DIRECTIONS[i] for i in top_indices]
        except Exception as e:
            return [f"Error suggesting research directions: {str(e)}"]

    def analyze_paper(self, file_content, file_name):
        """Main method to analyze a paper"""
        if file_name.endswith('.pdf'):
            text = self.extract_text_from_pdf(file_content)
        elif file_name.endswith('.docx'):
            text = self.extract_text_from_docx(file_content)
        elif file_name.endswith('.txt'):
            text = file_content.decode('utf-8')
        else:
            return {"error": "Unsupported file format"}
        
        # Extract sections
        future_work_text, limitations_text = self.extract_sections(text)
        
        # Create analysis results
        results = {
            "future_work": {
                "raw_text": future_work_text if future_work_text else "Not Found",
                "summary": self.summarize_to_bullets(future_work_text)
            },
            "limitations": {
                "raw_text": limitations_text if limitations_text else "Not Found",
                "summary": self.summarize_to_bullets(limitations_text)
            },
            "suggested_research_directions": self.suggest_research_directions(future_work_text, limitations_text)
        }
        
        return results