import os
import sys
from dotenv import load_dotenv

project_dir = os.path.dirname(os.path.dirname(__file__))

prefix_to_clear = "signify_"
keys_to_clear = [key for key in os.environ if key.startswith(prefix_to_clear)]
for key in keys_to_clear:
        del os.environ[key]

load_dotenv()

