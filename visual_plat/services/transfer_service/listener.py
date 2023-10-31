import iprocessing
import socket
import pickle


class Listener:
    def __init__(self, mode, callback):
        self.mode = mode
        self.callback = callback
        if mode == "process":
            # 初始化共享内存
            self.shared_memory = iprocessing.Value('i', 0)
        elif mode == "network":
            # 初始化 socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.bind(("localhost", 8000))
            self.sock.listen(5)

    def receive(self):
        if self.mode == "process":
            # 从共享内存中获取数据
            obj = pickle.loads(self.shared_memory.value)
        elif self.mode == "network":
            # 从 socket 中获取数据
            conn, addr = self.sock.accept()
            # 获取数据的长度
            data_length = int(conn.recv(1024).decode())
            # 根据数据的长度接收数据本身
            data = b''
            while len(data) < data_length:
                chunk = conn.recv(1024)
                if chunk == b'':
                    raise RuntimeError("socket connection broken")
                data += chunk
            obj = pickle.loads(data)
            conn.close()

        # 调用 callback 函数
        self.callback(obj)


class Notifier:
    def __init__(self, mode):
        self.mode = mode
        if mode == "process":
            # 初始化共享内存
            self.shared_memory = iprocessing.Value('i', 0)
        elif mode == "network":
            # 初始化 socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def send(self, obj):
        data = pickle.dumps(obj)
        if self.mode == "process":
            # 将数据写入共享内存
            self.shared_memory.value = data
        elif self.mode == "network":
            # 先发送数据的长度
            self.sock.connect(("localhost", 8000))
            self.sock.sendall(str(len(data)).encode())
            # 再发送数据本身
            self.sock.sendall(data)
            self.sock.close()
