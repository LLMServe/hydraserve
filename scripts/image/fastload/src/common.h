#pragma once

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <sys/time.h>
#include <unordered_map>
#include <fstream>
#include <sys/time.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <tuple>
#include <atomic>

#include <Python.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>

namespace st {

#define CHUNK_SIZE (1 << 24)    // 32MB

extern std::tuple<std::string, torch::Tensor> get_state_dict();
extern void load_dest_state_dict();
extern void delete_state_dict();
extern void start_load_shm(std::string shm_name, int64_t shm_addr, int64_t shm_size);
extern int8_t check_load_fin_shm();
extern std::tuple<std::string, torch::Tensor> get_state_dict_shm();
extern void send_tensors(std::vector<torch::Tensor> tensors, int64_t num_bytes_per_tensor, int64_t sock_fd);
extern void recv_copy_tensors(std::vector<torch::Tensor> origin_tensors, int64_t num_bytes_per_tensor, int64_t sock_fd);
extern int64_t check_copy_fin();

}