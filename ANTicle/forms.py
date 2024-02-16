from django import forms
class InputDataForm(forms.Form):
    Quelle_1 = forms.CharField(max_length=100)
    Quelle_2 = forms.CharField(max_length=100)
    Quelle_3 = forms.CharField(max_length=100)
    Quelle_4 = forms.CharField(max_length=100)
    Quelle_5 = forms.CharField(max_length=100)