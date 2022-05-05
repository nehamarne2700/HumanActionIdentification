from django.db import models
from .validators import file_size

# Create your models here.
class Video(models.Model):
    caption=models.CharField(max_length=100)
    video=models.FileField(upload_to="HAI/",null=True, verbose_name="",validators=[file_size])

    def __str__(self):
        return self.caption
