import socket


def find_free_port(master_addr):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((master_addr, 0))
    _, port = sock.getsockname()
    sock.close()
    return port
