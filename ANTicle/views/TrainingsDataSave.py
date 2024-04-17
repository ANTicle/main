from django.views import View
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import csv
import datetime
import os

@method_decorator(csrf_exempt, name='dispatch')
class CSVDataSaveView(View):
    def post(self, request, *args, **kwargs):
        headline = request.POST.get('headline')
        print(headline)
        text = request.POST.get('text')
        print(text)
        reaction = request.POST.get('reaction')
        print(reaction)
        file_exists = os.path.exists('training.csv')
        with open('training.csv', 'a' if file_exists else 'w', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Headline', 'Text', 'Reaction', 'Time'])
            writer.writerow([headline, text, reaction, datetime.datetime.now()])
        return JsonResponse({'status': 'saved'})

    def get(self, request, *args, **kwargs):
        # Redirect to a different view or raise an HTTP 405 Method Not Allowed error here
        pass