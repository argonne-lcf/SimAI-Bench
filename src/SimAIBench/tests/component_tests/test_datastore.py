from SimAIBench.datastore import DataStore, ServerManager
from typing import Dict
from SimAIBench import server_registry
from SimAIBench.config import ServerConfig
import multiprocessing as mp
import uuid
import os


def send(server_info: dict, return_dict: dict):
    ds = DataStore("store1", server_info=server_info)
    my_id = uuid.uuid4()
    ds.stage_write("key", my_id)
    return_dict["send"] = my_id

def recv(server_info: dict, return_dict: dict):
    ds = DataStore("store2", server_info=server_info)
    id = ds.stage_read("key")
    return_dict["recv"] = id

def test_datastore(server_config: ServerConfig):
    with mp.Manager() as manager:
        server_manager = ServerManager("server", config=server_config)
        server_manager.start_server()
        server_info = server_manager.get_server_info()
        
        return_dict = manager.dict()
        
        send_process = mp.Process(target=send, args=(server_info, return_dict))
        recv_process = mp.Process(target=recv, args=(server_info, return_dict))
        
        send_process.start()
        recv_process.start()
        
        send_process.join()
        recv_process.join()
        
        send_id = return_dict['send']
        recv_id = return_dict['recv']
        
        assert send_id == recv_id
    
    server_manager.stop_server()

if __name__ == "__main__":
    for server_type in ["filesystem","redis","dragon","node-local"]:
        try:
            if server_type == "redis":
                config = server_registry.create_config(type=server_type,redis_server_exe=os.environ.get("REDIS_SERVER_EXE","redis-server"))
            else:
                config = server_registry.create_config(type=server_type)
            test_datastore(config)
            print(f"{server_type} test passed!!")
            print("*"*50)
        except Exception as e:
            print(f"{server_type} test failed with exception {e}")
            if server_type == "dragon":
                print("Try using dragon <filename>")
            elif server_type == "redis":
                if str(e) == "redis_server_exe must be specified for Redis server":
                    print("Re-test after setting REDIS_SERVER_EXE=/path/to/redis-server environment variable")
            print("*"*50)