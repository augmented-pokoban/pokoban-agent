import subprocess
from threading import Thread
from time import sleep
from env import api


def _start_server():
    print("Starting pokoban-server")

    with subprocess.Popen(["java", "-jar", "pokoban-server-0.0.1.jar"],
                          stdout=subprocess.PIPE,
                          bufsize=1,
                          universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')

    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, p.args)


def start_server():
    server_thread = Thread(target=_start_server, daemon=True)
    server_thread.start()

    sleep(10)

    tries = 0
    while tries < 10:
        try:
            print('Pinging server... ', end='')
            api.ping_server()
            print('Success')
            return True
        except:
            print('Failure, sleeping for 10 seconds')
            sleep(10)

        tries += 1

    return False

if __name__ == "__main__":
    start_server()
