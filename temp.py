import requests

ip = "100.68.121.210"
port = 8290
id = "pomodoro_alert"

url = f"http://{ip}:{port}/trigger/{id}"


try:
    repose = requests.get(url)
    print("notication sent", reponse.status_code)
except Exception as e:
    print("error ", e)
