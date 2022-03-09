from server import getGloveData
import time

if __name__ == "__main__":
    while True:
        print("running")
        getGloveData()
        time.sleep(10)