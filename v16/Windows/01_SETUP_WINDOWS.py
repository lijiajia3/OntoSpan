"""Create the local Windows venv and install dependencies, as in v12."""
import argparse
from windows_launcher import setup


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--install-torch',action='store_true',help='兼容旧命令；现在01默认在本包.venv安装PyTorch')
    model_flags=ap.add_mutually_exclusive_group()
    model_flags.add_argument('--download',dest='download',action='store_true',help='默认：允许补齐两个backbone缓存')
    model_flags.add_argument('--offline-models',dest='download',action='store_false',help='只检查已有模型缓存，不下载模型；依赖安装仍需联网')
    ap.set_defaults(download=True)
    a=ap.parse_args()
    setup(download=a.download)


if __name__=='__main__':main()
