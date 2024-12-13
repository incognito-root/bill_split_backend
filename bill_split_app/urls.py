from django.urls import path
from .views import OCRAPIView

urlpatterns = [
    path('ocr/', OCRAPIView.as_view(), name='ocr'),
]
