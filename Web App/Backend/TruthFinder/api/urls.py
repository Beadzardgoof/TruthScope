from django.urls import path
from .views import upload_video

urlpatterns = [
    path('upload/', upload_video, name='upload-video'),
]
