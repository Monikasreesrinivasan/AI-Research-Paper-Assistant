# # services/paper_processor_service.py

# import os
# import re
# import json
# import time
# from typing import List, Dict, Any, Optional
# import PyPDF2
# import requests
# import io
# from groq import Groq
# from gtts import gTTS
# from pydantic import BaseModel

# class ResearchPaper(BaseModel):
#     title: str = ""
#     authors: List[str] = []
#     abstract: str = ""
#     content: str = ""
#     file_path: str = ""
#     summary: str = ""
#     topics: List[str] = []
#     summary_audio_path: str = ""
#     podcast_audio_path: str = ""
#     podcast_script: str = ""

# class PaperProcessor:
#     """Extract text from PDF and structure data"""
    
#     def extract_text_from_pdf(self, file_path: str) -> str:
#         """Extract text from a PDF file"""
#         with open(file_path, 'rb') as file:
#             reader = PyPDF2.PdfReader(file)
#             text = ""
#             for page in reader.pages:
#                 text += page.extract_text()
#         return text

#     def extract_metadata(self, text: str) -> tuple:
#         """Extract title, authors, and abstract from the text"""
#         # Simple extraction based on patterns - can be improved
#         title_match = re.search(r'^(.*?)\n', text)
#         title = title_match.group(1).strip() if title_match else "Unknown Title"

#         # Try to find authors - this is a simplified approach
#         author_section = re.search(r'(?:Author[s]?:|by)(.*?)(?:Abstract|Introduction)', text, re.IGNORECASE | re.DOTALL)
#         authors = []
#         if author_section:
#             author_text = author_section.group(1).strip()
#             # Split by common separators
#             for sep in [',', ';', 'and']:
#                 if sep in author_text:
#                     authors = [a.strip() for a in author_text.split(sep) if a.strip()]
#                     break
#             if not authors:
#                 authors = [author_text]

#         # Extract abstract
#         abstract_match = re.search(r'(?:Abstract|ABSTRACT)(.*?)(?:Introduction|Keywords|1\.)', text, re.IGNORECASE | re.DOTALL)
#         abstract = abstract_match.group(1).strip() if abstract_match else ""

#         return title, authors, abstract

#     def process_paper(self, file_path: str) -> Optional[ResearchPaper]:
#         """Process a paper from a file path"""
#         try:
#             text = self.extract_text_from_pdf(file_path)
#             title, authors, abstract = self.extract_metadata(text)

#             return ResearchPaper(
#                 title=title,
#                 authors=authors,
#                 abstract=abstract,
#                 content=text,
#                 file_path=file_path
#             )
#         except Exception as e:
#             print(f"Error processing paper {file_path}: {e}")
#             return None


# class PaperSummarizer:
#     """Use Groq LLM to summarize the paper"""
    
#     def __init__(self, api_key: str):
#         self.groq_client = Groq(api_key=api_key)

#     def summarize_paper(self, paper: ResearchPaper, model: str = "llama3-8b-8192") -> tuple:
#         """Summarize a paper using Groq LLM"""
#         try:
#             # Prepare the prompt for summarization
#             if paper.abstract:
#                 text_to_summarize = paper.abstract
#                 if len(text_to_summarize) > 8000:
#                     text_to_summarize = text_to_summarize[:8000]  # Limit token length
#             else:
#                 # If no abstract, use the first part of the content
#                 text_to_summarize = paper.content[:8000]

#             prompt = f"""
#             Please provide a comprehensive yet concise summary of the following research paper.
#             Focus on key contributions, methodology, results, and implications.
#             Structure the summary into these sections:
#             1. Main Research Question/Purpose
#             2. Methodology
#             3. Key Findings
#             4. Implications & Conclusion

#             Paper content:
#             {text_to_summarize}
#             """

#             # Call Groq API
#             response = self.groq_client.chat.completions.create(
#                 messages=[{"role": "user", "content": prompt}],
#                 model=model,
#                 max_tokens=2048
#             )

#             summary = response.choices[0].message.content
#             paper.summary = summary

#             # Extract topics from the summary
#             topic_prompt = f"""
#             Based on the following paper summary, list 3-5 key topics or research areas
#             that this paper belongs to. Return only the topics as a comma-separated list.

#             Summary:
#             {summary}
#             """

#             topic_response = self.groq_client.chat.completions.create(
#                 messages=[{"role": "user", "content": topic_prompt}],
#                 model=model,
#                 max_tokens=256
#             )

#             topics = topic_response.choices[0].message.content
#             paper.topics = [topic.strip() for topic in topics.split(',')]

#             return summary, paper.topics

#         except Exception as e:
#             print(f"Error summarizing paper {paper.title}: {e}")
#             paper.summary = f"Error generating summary: {str(e)}"
#             paper.topics = ["Uncategorized"]
#             return paper.summary, paper.topics


# class AudioGenerator:
#     """Generate audio from summary and create podcast"""
    
#     def __init__(self, output_dir: str = "audio_output"):
#         self.output_dir = output_dir
#         # Create output directory if it doesn't exist
#         os.makedirs(output_dir, exist_ok=True)

#     def generate_summary_audio(self, paper: ResearchPaper) -> Optional[str]:
#         """Generate audio from paper summary"""
#         try:
#             # Create a file name for the audio
#             file_name = re.sub(r'[^\w\s]', '', paper.title.lower())
#             file_name = re.sub(r'\s+', '_', file_name)
#             audio_path = f"{self.output_dir}/{file_name}_summary.mp3"

#             # Generate audio using gTTS
#             tts = gTTS(text=paper.summary, lang='en', slow=False)
#             tts.save(audio_path)

#             paper.summary_audio_path = audio_path
#             return audio_path
#         except Exception as e:
#             print(f"Error generating summary audio for {paper.title}: {e}")
#             return None

#     def generate_podcast(self, paper: ResearchPaper, groq_client: Groq) -> tuple:
#         """Generate a podcast discussing a single paper"""
#         try:
#             # Create podcast script using Groq
#             # Prepare podcast prompt with main topic from paper
#             main_topic = paper.topics[0] if paper.topics else "Research"
            
#             prompt = f"""
#             Create a podcast script discussing the following research paper on the topic of {main_topic}.
#             The podcast should:
#             1. Start with an introduction to the topic and why it matters
#             2. Present the paper's key findings, methodology, and contributions in an engaging way
#             3. Discuss the implications of this research for the field
#             4. End with thoughts on future research directions

#             Paper title: {paper.title}
#             Authors: {', '.join(paper.authors)}
#             Summary: {paper.summary}

#             Format the script with clear speaker parts (Host and Guest Expert) and natural, conversational language.
#             The podcast should feel like an interview between a host and a researcher discussing this paper.
#             """

#             response = groq_client.chat.completions.create(
#                 messages=[{"role": "user", "content": prompt}],
#                 model="llama3-8b-8192",
#                 max_tokens=3000
#             )

#             podcast_script = response.choices[0].message.content
#             paper.podcast_script = podcast_script

#             # Generate audio from the script
#             file_name = re.sub(r'[^\w\s]', '', paper.title.lower())
#             file_name = re.sub(r'\s+', '_', file_name)
#             podcast_path = f"{self.output_dir}/{file_name}_podcast.mp3"

#             tts = gTTS(text=podcast_script, lang='en', slow=False)
#             tts.save(podcast_path)

#             # Save the script as well
#             script_path = f"{self.output_dir}/{file_name}_podcast_script.txt"
#             with open(script_path, 'w') as f:
#                 f.write(podcast_script)

#             paper.podcast_audio_path = podcast_path
#             return podcast_path, podcast_script
#         except Exception as e:
#             print(f"Error generating podcast for {paper.title}: {e}")
#             return None, None


# class SinglePaperSystem:
#     """Main system to process a single paper"""
    
#     def __init__(self, groq_api_key: str, upload_dir: str = "uploads", audio_dir: str = "audio_output"):
#         self.processor = PaperProcessor()
#         self.summarizer = PaperSummarizer(api_key=groq_api_key)
#         self.audio_generator = AudioGenerator(output_dir=audio_dir)
#         self.upload_dir = upload_dir
#         self.audio_dir = audio_dir
        
#         # Create necessary directories
#         os.makedirs(upload_dir, exist_ok=True)
#         os.makedirs(audio_dir, exist_ok=True)

#     async def process_paper_file(self, file_path: str) -> Dict[str, Any]:
#         """Process a paper from a file path"""
#         # Process the paper
#         paper = self.processor.process_paper(file_path)
#         if not paper:
#             return {"error": "Failed to process paper"}
        
#         # Summarize paper
#         summary, topics = self.summarizer.summarize_paper(paper)
        
#         # Generate audio for the summary
#         summary_audio_path = self.audio_generator.generate_summary_audio(paper)
        
#         # Generate podcast for the paper
#         podcast_path, podcast_script = self.audio_generator.generate_podcast(paper, self.summarizer.groq_client)
        
#         # Create response
#         result = {
#             "title": paper.title,
#             "authors": paper.authors,
#             "abstract": paper.abstract,
#             "summary": paper.summary,
#             "topics": paper.topics,
#             "summary_audio": summary_audio_path.replace(self.audio_dir, "/audio") if summary_audio_path else None,
#             "podcast_audio": podcast_path.replace(self.audio_dir, "/audio") if podcast_path else None,
#             "podcast_script": podcast_script
#         }
        
#         return result


import os
import re
import json
import time
import PyPDF2
import requests
from groq import Groq
from gtts import gTTS
from typing import Dict, List, Optional, Tuple
from app.config import settings
from fastapi import UploadFile

# Groq API setup using the settings
GROQ_API_KEY = settings.GROQ_API_KEY

class ResearchPaper:
    def __init__(self, title="", authors=None, abstract="", content="", file_path=""):
        self.title = title
        self.authors = authors if authors is not None else []
        self.abstract = abstract
        self.content = content
        self.file_path = file_path
        self.summary = ""
        self.topics = []
        self.summary_audio_path = ""
        self.podcast_audio_path = ""
        self.podcast_script = ""

    def to_dict(self):
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "file_path": self.file_path,
            "summary": self.summary,
            "topics": self.topics,
            "summary_audio_path": self.summary_audio_path,
            "podcast_audio_path": self.podcast_audio_path,
            "podcast_script": self.podcast_script
        }

class PaperProcessor:
    def __init__(self):
        pass

    def extract_text_from_pdf(self, file_path):
        """Extract text from a PDF file"""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text

    def extract_metadata(self, text):
        """Extract title, authors, and abstract from the text"""
        # Simple extraction based on patterns - can be improved
        title_match = re.search(r'^(.*?)\n', text)
        title = title_match.group(1).strip() if title_match else "Unknown Title"

        # Try to find authors - this is a simplified approach
        author_section = re.search(r'(?:Author[s]?:|by)(.*?)(?:Abstract|Introduction)', text, re.IGNORECASE | re.DOTALL)
        authors = []
        if author_section:
            author_text = author_section.group(1).strip()
            # Split by common separators
            for sep in [',', ';', 'and']:
                if sep in author_text:
                    authors = [a.strip() for a in author_text.split(sep) if a.strip()]
                    break
            if not authors:
                authors = [author_text]

        # Extract abstract
        abstract_match = re.search(r'(?:Abstract|ABSTRACT)(.*?)(?:Introduction|Keywords|1\.)', text, re.IGNORECASE | re.DOTALL)
        abstract = abstract_match.group(1).strip() if abstract_match else ""

        return title, authors, abstract

    async def process_paper_from_upload(self, file: UploadFile) -> Optional[ResearchPaper]:
        """Process an uploaded paper file"""
        try:
            # Save the uploaded file temporarily
            temp_file_path = f"{settings.UPLOAD_DIR}/{file.filename}"
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Process the paper
            text = self.extract_text_from_pdf(temp_file_path)
            title, authors, abstract = self.extract_metadata(text)

            return ResearchPaper(
                title=title,
                authors=authors,
                abstract=abstract,
                content=text,
                file_path=temp_file_path
            )
        except Exception as e:
            print(f"Error processing paper: {e}")
            return None

class PaperSummarizer:
    def __init__(self):
        self.groq_client = Groq(api_key=GROQ_API_KEY)

    async def summarize_paper(self, paper: ResearchPaper, model="llama3-8b-8192"):
        """Summarize a paper using Groq LLM"""
        try:
            # Prepare the prompt for summarization
            if paper.abstract:
                text_to_summarize = paper.abstract
                if len(text_to_summarize) > 8000:
                    text_to_summarize = text_to_summarize[:8000]  # Limit token length
            else:
                # If no abstract, use the first part of the content
                text_to_summarize = paper.content[:8000]

            prompt = f"""
            Please provide a comprehensive yet concise summary of the following research paper.
            Focus on key contributions, methodology, results, and implications.
            Structure the summary into these sections:
            1. Main Research Question/Purpose
            2. Methodology
            3. Key Findings
            4. Implications & Conclusion

            Paper content:
            {text_to_summarize}
            """

            # Call Groq API
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=2048
            )

            summary = response.choices[0].message.content
            paper.summary = summary

            # Extract topics from the summary
            topic_prompt = f"""
            Based on the following paper summary, list 3-5 key topics or research areas
            that this paper belongs to. Return only the topics as a comma-separated list.

            Summary:
            {summary}
            """

            topic_response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": topic_prompt}],
                model=model,
                max_tokens=256
            )

            topics = topic_response.choices[0].message.content
            paper.topics = [topic.strip() for topic in topics.split(',')]

            return summary, paper.topics

        except Exception as e:
            print(f"Error summarizing paper {paper.title}: {e}")
            paper.summary = f"Error generating summary: {str(e)}"
            paper.topics = ["Uncategorized"]
            return paper.summary, paper.topics

class AudioGenerator:
    def __init__(self):
        self.output_dir = settings.AUDIO_OUTPUT_DIR
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    async def generate_summary_audio(self, paper: ResearchPaper):
        """Generate audio from paper summary"""
        try:
            # Create a file name for the audio
            file_name = re.sub(r'[^\w\s]', '', paper.title.lower())
            file_name = re.sub(r'\s+', '_', file_name)
            audio_path = f"{self.output_dir}/{file_name}_summary.mp3"

            # Generate audio using gTTS
            tts = gTTS(text=paper.summary, lang='en', slow=False)
            tts.save(audio_path)

            paper.summary_audio_path = audio_path
            return audio_path
        except Exception as e:
            print(f"Error generating summary audio for {paper.title}: {e}")
            return None

    async def generate_podcast(self, paper: ResearchPaper):
        """Generate a podcast discussing a single paper"""
        try:
            # Create podcast script using Groq
            groq_client = Groq(api_key=GROQ_API_KEY)

            # Prepare podcast prompt with main topic from paper
            main_topic = paper.topics[0] if paper.topics else "Research"
            
            prompt = f"""
            Create a podcast script discussing the following research paper on the topic of {main_topic}.
            The podcast should:
            1. Start with an introduction to the topic and why it matters
            2. Present the paper's key findings, methodology, and contributions in an engaging way
            3. Discuss the implications of this research for the field
            4. End with thoughts on future research directions

            Paper title: {paper.title}
            Authors: {', '.join(paper.authors)}
            Summary: {paper.summary}

            Format the script with clear speaker parts (Host and Guest Expert) and natural, conversational language.
            The podcast should feel like an interview between a host and a researcher discussing this paper.
            """

            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                max_tokens=3000
            )

            podcast_script = response.choices[0].message.content
            paper.podcast_script = podcast_script

            # Generate audio from the script
            file_name = re.sub(r'[^\w\s]', '', paper.title.lower())
            file_name = re.sub(r'\s+', '_', file_name)
            podcast_path = f"{self.output_dir}/{file_name}_podcast.mp3"

            tts = gTTS(text=podcast_script, lang='en', slow=False)
            tts.save(podcast_path)

            # Save the script as well
            script_path = f"{self.output_dir}/{file_name}_podcast_script.txt"
            with open(script_path, 'w') as f:
                f.write(podcast_script)

            paper.podcast_audio_path = podcast_path
            return podcast_path, podcast_script
        except Exception as e:
            print(f"Error generating podcast for {paper.title}: {e}")
            return None, None

class SinglePaperSystem:
    def __init__(self):
        self.processor = PaperProcessor()
        self.summarizer = PaperSummarizer()
        self.audio_generator = AudioGenerator()

    async def process_paper(self, file: UploadFile):
        """Process a single uploaded paper"""
        # Process the paper
        paper = await self.processor.process_paper_from_upload(file)
        if not paper:
            return None

        # Summarize paper
        summary, topics = await self.summarizer.summarize_paper(paper)
        
        # Generate audio for the summary
        summary_audio_path = await self.audio_generator.generate_summary_audio(paper)
        
        # Generate podcast for the paper
        podcast_path, podcast_script = await self.audio_generator.generate_podcast(paper)
        
        return paper