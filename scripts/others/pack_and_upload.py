"""
Zip the images and upload them to the server
"""
import argparse
import shutil
import os
import time
from scp import SCPClient
import paramiko
from multiprocessing import Pool


def arg_parser():
    parser = argparse.ArgumentParser(description='Pack and upload images to the server')
    parser.add_argument('--image_folder', help='Folder containing the images', required=True)
    parser.add_argument('--target_folder', help='Folder to save the zip file', required=True)
    parser.add_argument('--hostname', help='Server to upload the images to', required=True)
    parser.add_argument('--port', help='Port to connect to the server', default=22)
    parser.add_argument('--username', help='Username to connect to the server', required=True)
    parser.add_argument('--password', help='Password to connect to the server', default=None)
    parser.add_argument('--key_filename', help='Key file to connect to the server', default=None)
    parser.add_argument('--cache_folder', help='Folder to save the cache file')
    parser.add_argument('--processes', type=int, help='Number of processes to use', default=48)
    return parser


def pack_directory(directory_path, save_root):
    # 确保目录存在
    if not os.path.exists(directory_path):
        print(f"目录 {directory_path} 不存在")
        return
    output_filename = os.path.basename(directory_path)
    p = os.path.join(save_root, output_filename)+'.temp'
    # 创建压缩包
    if os.path.exists(p+'.zip'):
        print(f"已存在压缩包 {p}.zip")
    else:
        print(f"正在打包目录 {directory_path} 到 {p}")
        shutil.make_archive(p, 'zip', directory_path)
        os.rename(p+'.zip', os.path.join(save_root, output_filename)+'.zip')
        print(f"已成功打包目录 {directory_path} 到 {output_filename}.zip")
    return output_filename + '.zip'


def upload_file_scp(hostname, username, local_path, remote_path, password=None, key_filename=None, port=22):
    flag = None
    try:
        # 创建 SSH 客户端
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, port=port, username=username, password=password, key_filename=key_filename)

        # 创建 SCP 客户端
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(local_path, remote_path)
            print(f"文件 {local_path} 已上传到 {remote_path}")

    except Exception as e:
        flag = local_path
    finally:
        ssh.close()
    return flag


def upload(arg):
    sub_folder, cache_root, target_folder, index = arg
    print('Progress:', index)
    file_name = pack_directory(sub_folder, cache_root)
    sp = os.path.join(cache_root, file_name)
    tp = os.path.join(target_folder, file_name)
    
    # flag = upload_file_scp(HOSTNAME, USERNAME, sp, tp, password=PASSWORD, key_filename=KEY_FILENAME, port=PORT)
    # if flag is None:
    #     os.remove(sp)
    # else:
    #     print(f"上传失败: {flag}")
    

def main(args):
    image_folder = args.image_folder
    cache_folder = args.cache_folder
    target_folder = args.target_folder
    processes = args.processes
    print(cache_folder)
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    
    sub_folders = [os.path.join(image_folder, i) for i in os.listdir(image_folder)]
    cache_roots = [cache_folder] * len(sub_folders)
    target_roots = [target_folder] * len(sub_folders)
    args = list(zip(sub_folders, cache_roots, target_roots, list(range(len(sub_folders)))))
    # print(args)
    pool = Pool(processes=processes)
    pool.map(upload, args)

if __name__ == '__main__':    
    args = arg_parser().parse_args()
    # global HOSTNAME, PORT, USERNAME, PASSWORD, KEY_FILENAME
    HOSTNAME = args.hostname
    PORT = args.port
    USERNAME = args.username
    PASSWORD = args.password
    KEY_FILENAME = args.key_filename
    main(args)
