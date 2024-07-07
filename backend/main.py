from modules.sign_to_text import process_video
from modules.llm import construct_sentence
from modules.tts import text_to_speech
from scripts.json_to_list import json_conversion
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def generate_caption():
    try:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logger.info(f"Project directory: {project_dir}")

        results_file_path = os.path.join(project_dir, 'backend', 'video', 'results.json')
        logger.info(f"Results file path: {results_file_path}")
        

        # logger.info("Starting video processing...")
        # process_video()
        # logger.info("Video processing completed.")
        

        word_list = json_conversion(results_file_path)

        results = construct_sentence(word_list)
        logger.info(f"Constructed sentence: {results}")
        
        text_to_speech(results)
        return results


        

    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {fnf_error}")
        raise

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    generate_caption()



