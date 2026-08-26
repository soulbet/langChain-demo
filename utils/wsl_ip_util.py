import subprocess


class WslIp:
    def __init__(self):
        pass

    def get_wsl_ip(self):
        ip_output = subprocess.check_output("wsl hostname -I", shell=True).decode()
        wsl_ip = ip_output.strip().split()[0]
        return wsl_ip