import subprocess
from threading import Thread
from time import sleep


def start_server():
    print("Starting pokoban-server")

    with subprocess.Popen(["java", "-jar", "pokoban-server-0.0.1.jar"],
                          stdout=subprocess.PIPE,
                          bufsize=1,
                          universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')

    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, p.args)


if __name__ == "__main__":
    server_thread = Thread(target=start_server)
    server_thread.start()

    sleep(10)

    # TODO ping server on localhost:5000/api/pokoban/running

    # server_thread.join()  # close the server at some point?
