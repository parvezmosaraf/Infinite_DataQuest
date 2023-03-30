from django.db import models

class Contact (models.Model):
    name = models.CharField (max_length=99)
    phone = models.CharField (max_length=99)
    email = models.CharField (max_length=99)
    company = models.CharField (max_length=99)
    msg = models.TextField (max_length=99)

    def __str__(self):
        if len(self.name) > 50:
            return self.name[:50]+"..."
        return self.name


class ChatBot (models.Model):
    username = models.CharField(max_length=99, blank=True, null= True)
    name = models.CharField(max_length=99, blank=True, null= True)
    text = models.TextField(blank=True, null= True)
    date = models.DateField(auto_now_add=True)

    def __str__(self):
        if len(self.username) > 50:
            return self.username[:50]+"..."
        return self.username
