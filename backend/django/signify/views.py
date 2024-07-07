from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from rest_framework.decorators import api_view
from django.core.files.storage import default_storage
import logging

class VideoUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES['video']
        file_path = default_storage.save(file_obj.name, file_obj)
        return Response({"file_path": file_path}, status=status.HTTP_201_CREATED)
    

import sys
sys.path.append("..")
from main import generate_caption # Adjust the import path as needed

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
