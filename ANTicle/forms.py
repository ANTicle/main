from django import forms
class InputDataForm(forms.Form):
    Quelle_1 = forms.CharField(required=True, max_length=10000)
    Quelle_2 = forms.CharField(required=True, max_length=10000)
    Quelle_3 = forms.CharField(required=False, max_length=10000)
    Quelle_4 = forms.CharField(required=False, max_length=10000)
    Quelle_5 = forms.CharField(required=False, max_length=10000)