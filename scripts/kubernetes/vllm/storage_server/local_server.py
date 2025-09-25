import os
import re
import sys
import json
import time
import array
import traceback
import asyncio
import subprocess
import socket
import multiprocessing
from multiprocessing import shared_memory
runtime_start_time = time.time()

cached_state_dicts = {}
cached_state_dict_list = []
num_server = int(os.getenv("NUM_REMOTE_SERVER", "1"))
server_addr = os.getenv("REMOTE_SERVER_ADDR", "")
server_addr_2 = os.getenv("REMOTE_SERVER_ADDR_2", "")
use_cache = int(os.getenv("USE_CACHE", "0"))
slow_expr = int(os.getenv("SLOW_EXPR", "0"))
serverless_pilot = int(os.getenv("SERVERLESS_PILOT", "0"))
chunk_size = 32 * 1024 * 1024      # 32MB
background_tasks = set()

class SharedMemory:
    def __init__(self, size: int):
        self.shm = shared_memory.SharedMemory(create=True, size=size)
        blank_bytes = bytes(4096)
        for index in range(0, size, 4096):
            self.shm.buf[index:index+4096] = blank_bytes
        self.size = size
        self.blocks = [[0, size]]
        self.buffers = []
    
    def try_evict_cache(self):
        mn_last_used_time = 0
        mn_cache_id = -1
        mn_buffer_id = -1
        for id, cache_id in enumerate(self.buffers):
            cache = cached_state_dict_list[cache_id]
            if cache.num_serving_reqs == 0:
                last_used_time = cache.last_used_time
                if mn_cache_id == -1 or last_used_time < mn_last_used_time:
                    mn_last_used_time = last_used_time
                    mn_cache_id = cache_id
                    mn_buffer_id = id
        if mn_cache_id != -1:
            # evict the cache
            cache = cached_state_dict_list[mn_cache_id]
            print(f"Evict cache for {cache.req_header}, cache_id = {mn_cache_id}")
            cache.clear()
            del self.buffers[mn_buffer_id]
    
    async def allocate_buffer(self, size: int, cache_id: int, buffer: bool):
        while True:
            mx_block_id = -1
            mx_block_size = 0
            for index, block in enumerate(self.blocks):
                if block[1] - block[0] >= mx_block_size:
                    mx_block_size = block[1] - block[0]
                    mx_block_id = index
            if mx_block_size < size:
                await asyncio.sleep(0)
                self.try_evict_cache()
            else:
                break
        allocated_addr = self.blocks[mx_block_id][0]
        if mx_block_size == size:
            self.blocks.pop(mx_block_id)
        else:
            self.blocks[mx_block_id][0] += size
        if buffer:
            self.buffers.append(cache_id)
        print(f"allocate_buffer [{cache_id}]: ({allocated_addr}, {allocated_addr + size})")
        print(f"blocks = {self.blocks}")
        return allocated_addr

    def free_buffer(self, addr: int, size: int):
        print(f"free_buffer: ({addr}, {addr + size})")
        if len(self.blocks) == 0:
            self.blocks.append([addr, addr + size])
            return
        id = -1
        for index, block in enumerate(self.blocks):
            if block[0] >= addr:
                id = index
                break
        if id == -1:
            if addr == self.blocks[-1][1]:
                self.blocks[-1][1] += size
            else:
                self.blocks.append([addr, addr + size])
            return
        # insert
        if id > 0 and addr == self.blocks[id-1][1]:
            self.blocks[id-1][1] += size
            id -= 1
        else:
            self.blocks.insert(id, [addr, addr + size])
        # merge
        if id < len(self.blocks) - 1 and addr + size == self.blocks[id+1][0]:
            self.blocks[id+1][0] = self.blocks[id][0]
            self.blocks.pop(id)
    
    def free_all_buffers(self):
        for cache_id in self.buffers:
            cache = cached_state_dict_list[cache_id]
            if cache.num_serving_reqs != 0:
                print(f"Error: the cache to be freed still has {cache.num_serving_reqs} requests in serving, model = {cache.req_header['model']}")
            cache.clear()
        self.buffers = []

stime = time.time()

# Get maximum size of shared memory
if serverless_pilot == 1:
    max_shm_size_in_gb = 32
else:
    max_shm_size_in_gb = int(os.getenv('SHM_SIZE', '0'))
    if max_shm_size_in_gb == 0:
        # use half of shared memory
        cmd = "df -h | grep shm | awk 'NR==1{print $2}'"
        res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout
        res = res.decode('gbk')
        max_shm_size_in_gb = int(re.findall(r'\d+', res)[0]) - 2
shm_size = max_shm_size_in_gb * 1024 * 1024 * 1024
print(f"Start to initialize {max_shm_size_in_gb}GB shared memory.")
# Allocate almost maximum shared memory
while True:
    try:
        # delete all previous shared memory at first
        os.system("rm -rf /dev/shm/psm_*")
        time.sleep(1)
        model_memory = SharedMemory(shm_size)
        break
    except Exception as e:
        exc_info = sys.exc_info()
        print(f"Allocate shared memory gets exception: {e}")
        print("".join(traceback.format_exception(*exc_info)))
    # No enough memory, waiting...
    time.sleep(1)

print(f"Initialize shared memory elapsed {time.time() - stime} seconds.")

class CachedStateDict:
    NUM_THREAD = 2
    ID_COUNTER = 0
    USED_COUNTER = 0

    def __init__(self, req_header):
        self.cache_id = CachedStateDict.ID_COUNTER
        CachedStateDict.ID_COUNTER += 1
        self.num_server = num_server
        self.server_addr = server_addr
        self.server_addr_2 = server_addr_2
        self.chunk_size = chunk_size
        del req_header["pre_load"]
        self.req_header = req_header
        self.req_header["num_thread"] = str(CachedStateDict.NUM_THREAD)
        self.num_serving_reqs = 0
        self.has_model = False
        self.shm_created = False
        self.last_used_time = CachedStateDict.USED_COUNTER
        # For requests that do not have the whole model, we do not buffer it
        if use_cache == 0 or int(self.req_header["pp_size"]) > 1 or serverless_pilot == 1:
            self.buffer = False
        else:
            self.buffer = True
    
    def clear(self):
        self.shm_created = False
        self.has_model = False
        model_memory.free_buffer(self.shm_addr, self.shm_size)

    def process_header(self, req_header, query = False):
        if query:
            req_header["is_query"] = True
        req_header_bytes = json.dumps(req_header).encode('utf-8')
        length = len(req_header_bytes)
        length_bytes = length.to_bytes(length=8, byteorder='little', signed=False)
        return length_bytes + req_header_bytes
    
    def recv_part_content(self, thread_id, shm_name, start_pos, end_pos, recv_len_per_thread):
        stime = time.time()

        try:
            shm = shared_memory.SharedMemory(name=shm_name)
        except FileNotFoundError:
            print(f"Error: Shared Memory {shm_name} disappeared!")
            print(f"Current shm name = {model_memory.shm.name}")
            print(f"Current shm data = {model_memory.shm.buf[:16]}")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if self.num_server == 2 and thread_id >= CachedStateDict.NUM_THREAD / 2:
            server_addr = self.server_addr_2
        else:
            server_addr = self.server_addr
        sock.connect((server_addr, 8888))
        req_header = self.req_header
        req_header["thread_id"] = str(thread_id)
        sock.sendall(self.process_header(req_header))

        cur_recv_len = start_pos
        fin_flag = False        # this flag represents that all previous threads have finished downloading so that we can change the recv_len in shared memory now
        if thread_id == 0:
            fin_flag = True
        target_recv_len = recv_len_per_thread * thread_id + 8
        while True:
            new_len = sock.recv_into(shm.buf[cur_recv_len:cur_recv_len+self.chunk_size], nbytes=self.chunk_size)
            if new_len == 0:
                break
            cur_recv_len += new_len
            if fin_flag:
                cur_recv_len_ = cur_recv_len - self.shm_addr
                shm.buf[self.shm_addr:self.shm_addr+8] = cur_recv_len_.to_bytes(length=8, byteorder='little', signed=True)
            elif int.from_bytes(shm.buf[self.shm_addr:self.shm_addr+8], byteorder='little', signed=True) == target_recv_len:
                fin_flag = True
                cur_recv_len_ = cur_recv_len - self.shm_addr
                shm.buf[self.shm_addr:self.shm_addr+8] = cur_recv_len_.to_bytes(length=8, byteorder='little', signed=True)

            if cur_recv_len == end_pos:
                break
            if cur_recv_len > end_pos:
                print(f"Error: cur_recv_len = {cur_recv_len} is larger than end_pos = {end_pos}")
        
        sock.close()

    async def start_download_model(self):
        stime = time.time()
        if self.has_model:
            print(f"Error: there already has a download model process for model {self.req_header['model']}")
            return
        self.has_model = True

        print(f"Start download model for {self.req_header['model']}")

        reader, writer = await asyncio.open_connection(self.server_addr, 8888, limit=self.chunk_size)
        
        writer.write(self.process_header(self.req_header.copy(), query = True))
        await writer.drain()

        data_size_bytes = await reader.read(8)
        data_size = int.from_bytes(data_size_bytes, byteorder='little', signed=False)
        self.shm_size = data_size + 8
        self.shm_addr = await model_memory.allocate_buffer(self.shm_size, self.cache_id, self.buffer)
        cur_recv_len = 8
        model_memory.shm.buf[self.shm_addr:self.shm_addr+8] = cur_recv_len.to_bytes(length=8, byteorder='little', signed=True)
        self.shm_created = True

        # Process partition
        tasks = []
        num_thread = CachedStateDict.NUM_THREAD
        bytes_per_thread = data_size // num_thread
        for thread_id in range(0, num_thread):
            num_bytes_start = thread_id * bytes_per_thread + self.shm_addr + 8
            num_bytes_end = (thread_id + 1) * bytes_per_thread + self.shm_addr + 8
            if thread_id == num_thread - 1:
                num_bytes_end += data_size % num_thread
            p = multiprocessing.Process(target=self.recv_part_content, args=(thread_id, model_memory.shm.name, num_bytes_start, num_bytes_end, bytes_per_thread))
            tasks.append(p)
            p.start()

        while any(p.is_alive() for p in tasks):
            await asyncio.sleep(0.1)

        cur_recv_len = data_size + 8
        model_memory.shm.buf[self.shm_addr:self.shm_addr+8] = cur_recv_len.to_bytes(length=8, byteorder='little', signed=True)

        writer.close()
        await writer.wait_closed()

    async def return_model(self, reader, writer):
        self.num_serving_reqs += 1
        self.last_used_time = CachedStateDict.USED_COUNTER
        CachedStateDict.USED_COUNTER += 1

        if not self.has_model:
            # pre-load request has not arrived
            print(f"Pre-load request has not arrived, start a loading process for model {self.req_header['model']}")
            global background_tasks
            task = asyncio.create_task(self.start_download_model())
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

        while not self.shm_created:
            await asyncio.sleep(0.1)
        
        writer.write(self.shm_addr.to_bytes(length=8, byteorder='little', signed=True) + model_memory.size.to_bytes(length=8, byteorder='little', signed=True) + model_memory.shm.name.encode('utf-8'))
        await writer.drain()

        # wait for client reading data
        resp = await reader.read(50)

        self.num_serving_reqs -= 1

        if self.num_serving_reqs == 0:
            if not self.buffer:
                self.clear()

async def handler(reader, writer):
    try:
        global cached_state_dicts
        global cached_state_dict_list

        stime = time.time()

        req_header_size_bytes = await reader.read(8)
        req_header_size = int.from_bytes(req_header_size_bytes, byteorder='little', signed=False)

        if req_header_size == 0:
            # initialization query
            print('Check Initialization!')
            writer.write(bytes('initialization finished', 'utf-8'))
            await writer.drain()
            writer.close()
            return
        
        if req_header_size == 1:
            # free shared memory
            print('Free Memory!')
            model_memory.shm.close()
            model_memory.shm.unlink()
            writer.write(bytes('shared memory deleted', 'utf-8'))
            await writer.drain()
            writer.close()
            return
        
        if req_header_size == 2:
            # clear all model caches
            print('Clear Cache!')
            model_memory.free_all_buffers()
            # return total cache size
            shm_size_bytes = shm_size.to_bytes(length=8, byteorder='little', signed=False)
            writer.write(shm_size_bytes)
            await writer.drain()
            writer.close()
            return
        
        if req_header_size == 3:
            # query the existence of a model
            model_name = await reader.read(50)
            key = (model_name.decode('utf-8'), 0, 1, False)
            flag = 1
            if key not in cached_state_dicts:
                flag = 2
            else:
                cache = cached_state_dicts[key]
                if cache.has_model:
                    flag = 3
            writer.write(flag.to_bytes(length=8, byteorder='little', signed=False))
            await writer.drain()
            writer.close()
            return

        req_header_bytes = await reader.read(req_header_size)
        req_header = json.loads(req_header_bytes.decode("UTF-8"))
        model_id = req_header["model"]
        pp_rank = req_header["pp_rank"]
        pp_size = req_header["pp_size"]
        is_dest = req_header["is_dest"]
        pre_load = req_header["pre_load"]

        key = (model_id, pp_rank, pp_size, is_dest)
        if pre_load == "no":
            print(f"Received model request for {model_id} at {time.time() - runtime_start_time} s")
            if serverless_pilot == 1:
                if key not in cached_state_dicts:
                    newCachedStateDict = CachedStateDict(req_header)
                    cached_state_dicts[key] = newCachedStateDict
                    cached_state_dict_list.append(newCachedStateDict)
                    await newCachedStateDict.start_download_model()
                else:
                    newCachedStateDict = cached_state_dicts[key]
                    await newCachedStateDict.start_download_model()
            if key not in cached_state_dicts:
                # this happens when model request comes faster than pre-load request
                cachedStateDict = CachedStateDict(req_header)
                cached_state_dicts[key] = cachedStateDict
                cached_state_dict_list.append(cachedStateDict)
                await cachedStateDict.start_download_model()
            else:
                cachedStateDict = cached_state_dicts[key]
            await cachedStateDict.return_model(reader, writer)
            print(f"Process model request for {model_id} time cost = {time.time() - stime} seconds")
        else:
            print(f"Received pre-load request for {model_id} at {time.time() - runtime_start_time} s")
            if key not in cached_state_dicts:
                newCachedStateDict = CachedStateDict(req_header)
                cached_state_dicts[key] = newCachedStateDict
                cached_state_dict_list.append(newCachedStateDict)
            else:
                newCachedStateDict = cached_state_dicts[key]
            
            if not newCachedStateDict.has_model:
                await newCachedStateDict.start_download_model()
            print(f"Process pre-load request for {model_id} time cost = {time.time() - stime} seconds")
            
        writer.close()
    except Exception as e:
        exc_info = sys.exc_info()
        print(f"Handler gets exception: {e}")
        print("".join(traceback.format_exception(*exc_info)))

async def run_server():
    server = await asyncio.start_server(handler, '0.0.0.0', 6666)

    addr = server.sockets[0].getsockname()
    print('Serving on', addr)

    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(run_server())
