from django.urls import path
from .views import VideoUploadView
from .views import get_caption
from .views import get_audio
import logging

logger= logging.getLogger(__name__)

def log_path_caption(request):
    logger.info(f"Request path: {request.path}")
    return get_caption(request)

def log_path_audio(request):
    logger.info(f"Request path: {request.path}")
    return get_audio(request)

urlpatterns = [
    path('upload/', VideoUploadView.as_view(), name='video-upload'),
    path('caption/', log_path_caption, name='get_caption'), 
    path('audio/', log_path_audio, name='get_adio')
]


