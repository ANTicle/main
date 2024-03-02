from django import forms
class InputDataForm(forms.Form):
    Quelle_1 = forms.CharField(required=True, max_length=100000, widget=forms.Textarea(attrs={'class' : 'align-top'}))
    Quelle_2 = forms.CharField(required=True, max_length=100000, widget=forms.Textarea(attrs={'class' : 'align-top'}))
    Quelle_3 = forms.CharField(required=False, max_length=100000, widget=forms.Textarea(attrs={'class' : 'align-top'}))
    Quelle_4 = forms.CharField(required=False, max_length=100000, widget=forms.Textarea(attrs={'class' : 'align-top'}))
    Quelle_5 = forms.CharField(required=False, max_length=100000, widget=forms.Textarea(attrs={'class' : 'align-top'}))
    words = forms.IntegerField(required=True, max_value=5000, widget=forms.NumberInput)
    thema = forms.CharField(required=True, max_length=1000, widget=forms.Textarea(attrs={'class': 'align-top'}))

