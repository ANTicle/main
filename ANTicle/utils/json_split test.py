import json


def split_json_string(json_obj):


    # Separate Quelle keys
    quelle_dict = {k: v for k, v in json_obj.items() if k.startswith('Quelle')}

    with open('quelle.json', 'w') as file:
        json.dump(quelle_dict, file)

    # Put the remaining keys in another dictionary
    remaining_dict = {k: v for k, v in json_obj.items() if not k.startswith('Quelle') and k != 'csrfmiddlewaretoken'}

    with open('remaining.json', 'w') as file:
        json.dump(remaining_dict, file)

    return quelle_dict, remaining_dict


# test
json_obj = {"csrfmiddlewaretoken": "piplT9zMwOkODu5GdcSmDAeLl44UqmdxpXdwQaHJHPfacF7bMLkv5qr0Wmbn2AJ4", "Quelle_1": " Bis zum Abend zeitweise Windböen, in der Nacht im Herzogtun Lauenburg vereinzelt leichter Frost\r\n\r\nHeute Mittag und im weiteren Tagesverlauf Wechsel aus Sonne und Wolken, später an der Ostsee vereinzelt ein paar Regentropfen, ansonsten niederschlagsfrei. Bei Höchstwerten zwischen 5 Grad an der Ostseeküste und 8 Grad in Hamburg leichter Temperaturrückgang zu den Vortagen. Mäßiger, an den Küsten teils starker und böiger Ost- bis Nordostwind.\r\n\r\nIn der Nacht zum Mittwoch überwiegend stark bewölkt oder bedeckt, bevorzugt in der Nordosthälfte nordwestwärts ziehender Regen. Tiefstwerte um 1 Grad, auf Helgoland 4 Grad. Meist schwacher, an den Küsten mäßiger bis frischer nordöstlicher Wind.\r\n", "Quelle_2": "Vorhersage - morgen\r\n\r\nAm Mittwoch zunächst vielfach dicht bewölkt, nach Norden hin örtlich etwas Regen, im weiteren Tagesverlauf Auflockerungen. Bis zu 5 Grad auf Fehmarn und 9 Grad in Hamburg. Schwacher bis mäßiger Ost- bis Nordostwind. \r\n\r\nIn der Nacht zum Donnerstag meist wolkig oder stark bewölkt, später auch trüb, meist trocken. Tiefstwerte im Binnenland zwischen 0 und 2 Grad, auf Helgoland 4 Grad. Schwacher, an der Ostsee stellenweise auch mäßiger Ost- bis Nordostwind. \r\n", "Quelle_3": "", "Quelle_4": "", "Quelle_5": "", "words": "250", "thema": "test 2"}

quelle, other = split_json_string(json_obj)
print("Quelle: ", quelle)
print("Other: ", other)