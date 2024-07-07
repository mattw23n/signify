from django.urls import path
from .views import VideoUploadView
from .views import get_caption
import logging

logger= logging.getLogger(__name__)

def log_path(request):
    logger.info(f"Request path: {request.path}")
    return get_caption(request)

urlpatterns = [
    path('upload/', VideoUploadView.as_view(), name='video-upload'),
    path('caption/', log_path, name='get_caption'),  # Add logging to the URL pattern
]

# urlpatterns = [
#     path('upload/', VideoUploadView.as_view(), name='video-upload'),
#     path('caption/', get_caption, name='get_caption'),
# ]
