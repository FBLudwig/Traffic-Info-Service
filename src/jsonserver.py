from bottle import route, run

predictions = None

@route('/traffic-info')
def json_response():
    ka_nord = ""
    ka_mitte = ""
    ettlingen = ""
    try:
        ka_nord = int((predictions["Karlsruhe-Nord-A"] + predictions["Karlsruhe-Nord-B"]) / 2)
        ka_mitte = int((predictions["Karlsruhe-Mitte-A"] + predictions["Karlsruhe-Mitte-B"]) / 2)
        ettlingen = int((predictions["Ettlingen-A"] + predictions["Ettlingen-B"]) / 2)
    except:
        print("Value not in predictions")
    return { "Karlsruhe Nord": ka_nord, "Karlsruhe Mitte": ka_mitte, "Ettlingen": ettlingen}


def start(pred):
    global predictions
    predictions = pred
    run(host='0.0.0.0', port=8079, server='cherrypy')
