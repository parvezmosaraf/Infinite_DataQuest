# Generated by Django 4.1.6 on 2023-03-24 18:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0002_rename_message_contact_msg'),
    ]

    operations = [
        migrations.CreateModel(
            name='ChatBot',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(blank=True, max_length=99, null=True)),
                ('name', models.CharField(blank=True, max_length=99, null=True)),
                ('text', models.TextField(blank=True, null=True)),
                ('date', models.DateField(auto_now_add=True)),
            ],
        ),
    ]
