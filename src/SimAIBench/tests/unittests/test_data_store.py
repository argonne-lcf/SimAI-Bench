from SimAIBench.component import ServerManager, DataStore
import os

def test_daos(config:dict={"type":"daos","mode":"posix","server-address":f"/tmp/{os.getenv('USER')}/daos_mount"}):
    server = ServerManager("daos_server",config=config)
    server_info = server.get_server_info()

    ds = DataStore("daos_ds",server_info=server_info)

    ds.stage_write("key","value")
    value=ds.stage_read("key")
    assert value=="value",f"Expected 'value', but got {value}"


if __name__ == "__main__":
    # test_daos()
    test_daos({"type":"daos","mode":"kv","pool_label":"datascience","container_label":"test_daos"})
    print("All tests finished successfully")

