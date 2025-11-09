"""
Gunicorn 配置文件 - 多进程多GPU部署

启动方式：
    CUDA_VISIBLE_DEVICES=0,1 python server.py --workers 4
"""
import os

# 服务器配置
bind = "0.0.0.0:8020"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"

# 单线程模式（避免线程安全问题）
# worker_connections = 1
# threads = 1

# Worker 生命周期
max_requests = 1000
max_requests_jitter = 100
graceful_timeout = 30
timeout = 300
keepalive = 5

# 日志
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)sµs'

# 不要预加载应用（避免CUDA上下文问题）
preload_app = False


def get_available_gpus():
    """从 CUDA_VISIBLE_DEVICES 获取 GPU 列表"""
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        return [gpu.strip() for gpu in cuda_visible.split(',') if gpu.strip()]
    return []


def post_fork(server, worker):
    """
    Worker fork 后执行，为每个 worker 分配单独的 GPU
    实现方式：重新设置 CUDA_VISIBLE_DEVICES 为单个 GPU
    """
    import sys
    
    available_gpus = get_available_gpus()
    
    if available_gpus:
        worker_idx = worker.age - 1
        assigned_gpu = available_gpus[worker_idx % len(available_gpus)]
        os.environ['CUDA_VISIBLE_DEVICES'] = assigned_gpu
        print(f"[Gunicorn] Worker {worker.age} (PID: {worker.pid}) -> GPU {assigned_gpu}", file=sys.stderr)
    else:
        print(f"[Gunicorn] Worker {worker.age} (PID: {worker.pid}) -> default device", file=sys.stderr)
    
    os.environ['WORKER_ID'] = str(worker.age)


def worker_int(worker):
    """Worker 收到 INT 信号时调用"""
    print(f"[Gunicorn] Worker {worker.pid} received INT signal, shutting down...")


def worker_abort(worker):
    """Worker 异常退出时调用"""
    print(f"[Gunicorn] Worker {worker.pid} aborted!")


def on_exit(server):
    """服务器退出时调用"""
    print("[Gunicorn] Server shutting down...")

