from rest_framework import serializers
from .models import YourModel  # Replace with your model

class YourModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = YourModel
        fields = '__all__'
