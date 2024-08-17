import schedule
import time

def job():
    print("hello world")

schedule.every().day.at("14:37").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
    