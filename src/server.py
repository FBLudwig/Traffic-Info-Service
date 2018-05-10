import sys
import urllib
from multiprocessing import Process, Manager

import jsonserver
import test

cameras = [
    ("Karlsruhe-Nord-A", "https://www.svz-bw.de/kamera/ftpdata/KA031/KA031_gross.jpg?1525809142"),
    ("Karlsruhe-Nord-B", "https://www.svz-bw.de/kamera/ftpdata/KA032/KA032_gross.jpg?1525809471"),
    ("Karlsruhe-Mitte-A", "https://www.svz-bw.de/kamera/ftpdata/KA041/KA041_gross.jpg?1525809496"),
    ("Karlsruhe-Mitte-B", "https://www.svz-bw.de/kamera/ftpdata/KA042/KA042_gross.jpg?1525809515"),
    ("Ettlingen-A", "https://www.svz-bw.de/kamera/ftpdata/KA061/KA061_gross.jpg?1525809536"),
    ("Ettlingen-B", "https://www.svz-bw.de/kamera/ftpdata/KA062/KA062_gross.jpg?1525809551"),
]


def retrieve_image(url, path_to_save_location):
    opener = urllib.request.build_opener()
    opener.addheaders = [("Referer", "https://www.svz-bw.de")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, path_to_save_location)


def start_json_server():
    print("Starting jsonserver")
    jsonserver.start(predictions)
    print("finished jsonserver")


def count_cars(camera_name, url, predictions):
    while True:
        path_to_save_location = "data/A5-webcams/images/{}.jpg".format(camera_name)
        retrieve_image(url, path_to_save_location)
        print("{} saved".format(camera_name))
        prediction = test.main(sys.argv[1:], camera_name + ".jpg")
        print("{} | Prediction: {} cars".format(camera_name, prediction))
        predictions[camera_name] = prediction
        print(predictions)



if __name__ == '__main__':
    with Manager() as manager:
        predictions = manager.dict()

        for camera in cameras:
            camera_name = camera[0]
            url = camera[1]
            process = Process(target=count_cars, args=(camera_name, url, predictions))
            process.start()

        print("Processes started")

        server_process = Process(target=start_json_server)
        server_process.start()
        server_process.join()


