from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from django.conf import settings
from django.http import FileResponse
from rest_framework.decorators import api_view
from django.core.files.storage import default_storage
import logging
import os

class VideoUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES['video']
        file_path = default_storage.save(file_obj.name, file_obj)
        return Response({"file_path": file_path}, status=status.HTTP_201_CREATED)
    

import sys
sys.path.append("..")
from main import generate_caption


logger = logging.getLogger(__name__)

@api_view(['GET'])
def get_caption(request):
    try:
        results = generate_caption()
        return Response({"results": results})
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {fnf_error}")
        return Response({"error": "File not found"}, status=404)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return Response({"error": "An error occurred"}, status=500)
    
@api_view(['GET'])
def get_audio(request):
    try:
        audio_path = os.path.join(settings.BASE_DIR, 'resources', 'output.mp3')
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File not found: {audio_path}")
        
        return FileResponse(open(audio_path, 'rb'), content_type='audio/mpeg')
        
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {fnf_error}")
        return Response({"error": "File not found"}, status=404)
    
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return Response({"error": "An error occurred"}, status=500)
