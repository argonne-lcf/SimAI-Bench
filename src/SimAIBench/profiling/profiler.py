from .timer import Timer
from SimAIBench.datastore import DataStore
from SimAIBench.dag import Callable, DagStore
from typing import Dict, List, Union
import logging
import os
import uuid
import json
import cloudpickle

class Profiler:
    def __init__(self,server_info: Dict):
        self.server_info = server_info
        self.store = DataStore(f"profiler-{str(uuid.uuid4())}",server_info=server_info)

    def __setstate__(self, state):
        self.__dict__.update(state)


class DagStoreProfiler(Profiler):
    def __init__(self, datastore: DagStore, server_info: Dict):
        super().__init__(server_info)
        self._datastore = datastore
        self.durations: Dict[str, List] = {}
        self.id = str(uuid.uuid4())
    
    def _wrap_method(self, method_name: str):
        """Wrap a method to time its execution."""
        original_method = getattr(self._datastore, method_name)
        def wrapped(*args, **kwargs):
            with Timer(f"{method_name}") as timer:
                result = original_method(*args, **kwargs)
        
            try:
                self.durations[method_name].append(timer.get_duration())
            except KeyError:
                self.durations[method_name] = [timer.get_duration()]
            
            self.store.stage_write(f"dagstore-{method_name}-{self.id}-{len(self.durations[method_name])-1}",
                                   timer.get_duration())
            
            return result
        return wrapped

    def __getattr__(self, method_name: str):
        # logger.debug(f"getting attribute for {method_name}"
        try:
            return object.__getattribute__(self,method_name)
        except AttributeError:
            attr = getattr(self.__dict__['_datastore'], method_name)
            if callable(attr):
                wrapped = self._wrap_method(method_name)
                return wrapped
            return attr
    
    def report(self):
        tmp_dir = os.environ.get("SIMAIBENCH_PROFILER_TMPDIR",
                                 os.path.join(os.getcwd(), "./.profiler_tmp","dagstore"))
        os.makedirs(tmp_dir,exist_ok=True)
        dump = {}
        for method, times in self.durations.items():
            if times:
                avg = sum(times) / len(times)
                min_t = min(times)
                max_t = max(times)
                total = sum(times)
                count = len(times)
                dump[method] = {
                                'avg': avg,
                                'min': min_t,
                                'max': max_t,
                                'total': total,
                                'count': count
                                }
        if len(dump) > 0:
            with open(os.path.join(tmp_dir, f"{str(uuid.uuid4())}.json"), 'w') as f:
                json.dump(dump, f, indent=4)

    
    def copy(self):
        if isinstance(self._datastore, DagStore):
            new_dag_store = self._datastore.copy()
            return DagStoreProfiler(new_dag_store,self.server_info)
        else:
            raise AttributeError

    def __repr__(self):
        return f"<DataStoreProfiler wrapping {self._datastore}>"
    
class CallableProfiler(Profiler):
    def __init__(self, callable: Callable, server_info:Dict):
        super().__init__(server_info)
        self._callable = callable
        self.durations: Dict[str: List] = {}
        self.__name__ = callable.__name__
        self.id = str(uuid.uuid4())

    @property
    def callable(self):
        return self._callable
    
    def __call__(self, *args, **kwargs):
        with Timer(f"{self.__name__}") as timer:
            result =  self._callable(*args, **kwargs)
        method_name = self.__name__
        try:
            self.durations[method_name].append(timer.get_duration())
        except KeyError:
            self.durations[method_name] = [timer.get_duration()]
        
        self.store.stage_write(f"callable-{method_name}-{self.id}-{len(self.durations[method_name])-1}",
                                   timer.get_duration())
        return result
    
    def report(self):

        tmp_dir = os.environ.get("SIMAIBENCH_PROFILER_TMPDIR",
                                 os.path.join(os.getcwd(), "./.profiler_tmp","callables"))
        os.makedirs(tmp_dir,exist_ok=True)
        dump = {}
        for method, times in self.durations.items():
            if times:
                avg = sum(times) / len(times)
                min_t = min(times)
                max_t = max(times)
                total = sum(times)
                count = len(times)

                dump[method] = {
                    'avg': avg,
                    'min': min_t,
                    'max': max_t,
                    'total': total,
                    'count': count
                }
        if len(dump) > 0:
            with open(os.path.join(tmp_dir, f"{self.__name__ + '-' + str(uuid.uuid4())}.json"), 'w') as f:
                json.dump(dump, f,indent=4)
    
    def __repr__(self):
        return f"<CallableProfiler wrapping {self._callable}>"
    
    def __del__(self):
        self.report()

class DataStoreProfiler(Profiler):
    def __init__(self, datastore: DataStore, server_info: Dict):
        super().__init__(server_info)
        self._datastore = datastore
        self.durations: Dict[str, List] = {}
        self.id = str(uuid.uuid4())
    
    def _wrap_method(self, method_name: str):
        """Wrap a method to time its execution."""
        original_method = getattr(self._datastore, method_name)
        def wrapped(*args, **kwargs):
            with Timer(f"{method_name}") as timer:
                result = original_method(*args, **kwargs)
        
            try:
                self.durations[method_name].append(timer.get_duration())
            except KeyError:
                self.durations[method_name] = [timer.get_duration()]

            if method_name == "stage_read":
                nbytes = len(cloudpickle.dumps(result))
            elif method_name == "stage_write":
                nbytes = len(cloudpickle.dumps(args[1]))
            else:
                nbytes = 0
                
            self.store.stage_write(f"datastore-{method_name}-{self.id}-{len(self.durations[method_name])-1}",
                                   (timer.get_duration(),nbytes))
            
            return result
        return wrapped

    def __getattr__(self, method_name: str):
        # logger.debug(f"getting attribute for {method_name}"
        try:
            return object.__getattribute__(self,method_name)
        except AttributeError:
            attr = getattr(self.__dict__['_datastore'], method_name)
            if callable(attr):
                wrapped = self._wrap_method(method_name)
                return wrapped
            return attr
    
    def report(self):
        tmp_dir = os.environ.get("SIMAIBENCH_PROFILER_TMPDIR",
                                 os.path.join(os.getcwd(), "./.profiler_tmp","dagstore"))
        os.makedirs(tmp_dir,exist_ok=True)
        dump = {}
        for method, times in self.durations.items():
            if times:
                avg = sum(times) / len(times)
                min_t = min(times)
                max_t = max(times)
                total = sum(times)
                count = len(times)
                dump[method] = {
                                'avg': avg,
                                'min': min_t,
                                'max': max_t,
                                'total': total,
                                'count': count
                                }
        if len(dump) > 0:
            with open(os.path.join(tmp_dir, f"{str(uuid.uuid4())}.json"), 'w') as f:
                json.dump(dump, f, indent=4)

    
    def copy(self):
        if isinstance(self._datastore, DagStore):
            new_dag_store = self._datastore.copy()
            return DagStoreProfiler(new_dag_store,self.server_info)
        else:
            raise AttributeError

    def __repr__(self):
        return f"<DataStoreProfiler wrapping {self._datastore}>"