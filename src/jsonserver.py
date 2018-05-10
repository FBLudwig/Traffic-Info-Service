from bottle import route, run

predictions = None

@route('/traffic-info')
def json_response():
    ka_nord = ""
    ka_mitte = ""
    ettlingen = ""
    try:
        ka_nord = predictions["Karlsruhe-Nord-A"]
        ka_mitte = predictions["Karlsruhe-Mitte-A"]
        ettlingen = predictions["Ettlingen-A"]
    except:
        print("Exception - Wert nicht in predictions")
    return { "Karlsruhe Nord": ka_nord, "Karlsruhe Mitte": ka_mitte, "Ettlingen": ettlingen}


def start(pred):
    global predictions
    predictions = pred
    run(host='0.0.0.0', port=8079)
