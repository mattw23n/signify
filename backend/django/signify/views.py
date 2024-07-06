from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from django.core.files.storage import default_storage

class VideoUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES['video']
        file_path = default_storage.save(file_obj.name, file_obj)
        return Response({"file_path": file_path}, status=status.HTTP_201_CREATED)
