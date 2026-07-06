#!/usr/bin/env python
"""统一启动和关闭康复评估系统前后端"""
import subprocess
import sys
import os
import time
import argparse
import shutil
import urllib.request
import urllib.error
import socket

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def is_port_available(port, host='127.0.0.1'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


class SystemManager:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None

    def start_backend(self):
        """启动后端服务"""
        if not is_port_available(8000):
            raise RuntimeError("端口 8000 已被占用。请先关闭已有后端服务，或直接访问已运行的 http://localhost:8000")

        print("[启动] 后端服务 (FastAPI on port 8000)...")
        backend_dir = os.path.join(PROJECT_ROOT, 'backend')
        self.backend_process = subprocess.Popen(
            [sys.executable, 'main.py'],
            cwd=backend_dir
        )
        print(f"[后端] PID: {self.backend_process.pid}")

    def wait_for_backend(self, timeout=30):
        """等待后端健康检查通过"""
        deadline = time.time() + timeout
        health_url = 'http://localhost:8000/api/health'
        while time.time() < deadline:
            if self.backend_process and self.backend_process.poll() is not None:
                print("[错误] 后端进程已退出")
                return False
            try:
                with urllib.request.urlopen(health_url, timeout=1) as response:
                    if response.status == 200:
                        print("[后端] 健康检查通过")
                        return True
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
        print("[警告] 后端健康检查超时")
        return False

    def start_frontend(self):
        """启动前端服务"""
        if not is_port_available(5173):
            raise RuntimeError("端口 5173 已被占用。请先关闭已有前端服务，或直接访问已运行的 http://localhost:5173")

        print("[启动] 前端服务 (Vite dev server on port 5173)...")
        frontend_dir = os.path.join(PROJECT_ROOT, 'frontend')
        npm_cmd = shutil.which('npm') or shutil.which('npm.cmd')
        if not npm_cmd:
            raise RuntimeError("npm not found in PATH")
        self.frontend_process = subprocess.Popen(
            [npm_cmd, 'run', 'dev', '--', '--port', '5173', '--strictPort'],
            cwd=frontend_dir
        )
        print(f"[前端] PID: {self.frontend_process.pid}")

    def stop(self):
        """停止所有服务"""
        print("\n[关闭] 正在停止服务...")

        if self.backend_process:
            print(f"[后端] 终止进程 {self.backend_process.pid}...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
                self.backend_process.wait()

        if self.frontend_process:
            print(f"[前端] 终止进程 {self.frontend_process.pid}...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
                self.frontend_process.wait()

        print("[关闭] 所有服务已停止")

    def run(self, backend_only=False, frontend_only=False):
        """运行服务"""
        exit_code = 0
        try:
            if not frontend_only:
                self.start_backend()
                if not self.wait_for_backend():
                    raise RuntimeError("后端健康检查失败，停止启动流程")
            if not backend_only:
                self.start_frontend()

            print("\n" + "="*50)
            print("康复评估系统已启动")
            print("  后端: http://localhost:8000")
            print("  前端: http://localhost:5173")
            print("  按 Ctrl+C 停止服务")
            print("="*50 + "\n")

            # 保持运行
            while True:
                # 检查进程状态
                if self.backend_process and self.backend_process.poll() is not None:
                    print("[错误] 后端进程意外退出!")
                    break
                if self.frontend_process and self.frontend_process.poll() is not None:
                    print("[错误] 前端进程意外退出!")
                    break
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n[收到] 停止信号")
        except Exception as e:
            exit_code = 1
            print(f"[错误] {e}")
        finally:
            self.stop()
        return exit_code


def main():
    parser = argparse.ArgumentParser(description='康复评估系统管理脚本')
    parser.add_argument('--backend-only', action='store_true', help='只启动后端')
    parser.add_argument('--frontend-only', action='store_true', help='只启动前端')
    args = parser.parse_args()

    manager = SystemManager()
    sys.exit(manager.run(backend_only=args.backend_only, frontend_only=args.frontend_only))


if __name__ == '__main__':
    main()
