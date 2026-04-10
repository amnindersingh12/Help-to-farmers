import multiprocessing

bind = "127.0.0.1:5002"
workers = max(2, multiprocessing.cpu_count() // 2)
worker_class = "gthread"
threads = 4
timeout = 120
graceful_timeout = 30
keepalive = 5
max_requests = 1000
max_requests_jitter = 100
accesslog = "-"
errorlog = "-"
loglevel = "info"
