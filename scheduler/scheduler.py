# Trading Bot V3 - scheduler/scheduler.py
# Task scheduler for periodic operations

import threading
import time
from utils.logger import get_logger

logger = get_logger("scheduler")

class TaskScheduler:
    def __init__(self):
        self.tasks = []
        self.running = False
    
    def add_interval(self, name: str, func, interval_seconds: int):
        self.tasks.append({
            "name": name,
            "func": func,
            "interval": interval_seconds,
            "last_run": 0
        })
        logger.info(f"Scheduled {name} every {interval_seconds}s")
    
    def _loop(self):
        while self.running:
            now = time.time()
            for task in self.tasks:
                if now - task["last_run"] >= task["interval"]:
                    try:
                        task["func"]()
                    except Exception as e:
                        logger.error(f"Task {task['name']} error: {e}")
                    task["last_run"] = now
            time.sleep(1)
    
    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info("Scheduler started")
    
    def stop(self):
        self.running = False

# Convenience function
def run_with_scheduler(main_func, cycle_interval: int, 
                       extra_tasks: dict = None):
    scheduler = TaskScheduler()
    scheduler.add_interval("main_cycle", main_func, cycle_interval)
    
    if extra_tasks:
        for name, (func, interval) in extra_tasks.items():
            scheduler.add_interval(name, func, interval)
    
    scheduler.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scheduler.stop()
