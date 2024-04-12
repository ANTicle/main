from django.views import View
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import csv

@method_decorator(csrf_exempt, name='dispatch')
class CSVDataSaveView(View):
    def post(self, request, *args, **kwargs):
        headline = request.POST.get('headline')
        print(headline)
        text = request.POST.get('text')
        print(text)
        reaction = request.POST.get('reaction')
        print(reaction)
        with open('training.csv', mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(['Headline', 'Text', 'Reaction'])
            writer.writerow([headline, text, reaction])
        return JsonResponse({'status': 'saved'})

    def get(self, request, *args, **kwargs):
        # Redirect to a different view or raise an HTTP 405 Method Not Allowed error here
        pass