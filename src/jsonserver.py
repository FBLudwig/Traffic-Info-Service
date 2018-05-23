from bottle import route, run
import server

predictions = None

@route('/traffic-info')
def json_response():
    ka_nord = ""
    ka_mitte = ""
    ettlingen = ""
    try:
        # Calculate traffic density in percent
        ka_nord = int((predictions["Karlsruhe-Nord-A"] / server.camera_max_cars["Karlsruhe-Nord-A"] + predictions["Karlsruhe-Nord-B"] / server.camera_max_cars["Karlsruhe-Nord-B"]) / 2 * 100)
        ka_mitte = int((predictions["Karlsruhe-Mitte-A"] / server.camera_max_cars["Karlsruhe-Mitte-A"] + predictions["Karlsruhe-Mitte-B"] / server.camera_max_cars["Karlsruhe-Mitte-B"]) / 2 * 100)
        ettlingen = int((predictions["Ettlingen-A"] / server.camera_max_cars["Ettlingen-A"] + predictions["Ettlingen-B"] / server.camera_max_cars["Ettlingen-B"]) / 2 * 100)
    except:
        print("Value not in predictions")
    return { "Karlsruhe Nord": ka_nord, "Karlsruhe Mitte": ka_mitte, "Ettlingen": ettlingen}


def start(pred):
    global predictions  # Variable shared between processes
    predictions = pred
    run(host='0.0.0.0', port=8079, server='cherrypy')
