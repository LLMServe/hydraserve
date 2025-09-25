#include "common.h"
#include "json.hpp"
#include "cuda_utils.h"
#include <c10/cuda/CUDACachingAllocator.h>

using json = nlohmann::json;

namespace st {

torch::ScalarType getDtype(std::string dtype) {
    if (dtype == "BF16") return torch::kBFloat16;
    if (dtype == "F16") return torch::kFloat16;
    if (dtype == "F32") return torch::kFloat32;
    if (dtype == "F64") return torch::kFloat64;
    throw std::runtime_error("Unrecognized dtype.");
}

static char *myBuffer = NULL;
volatile static int64_t *cur_recv_end = NULL;
static std::vector<std::string> tensor_name_list;
static std::vector<torch::Tensor> tensor_list;
static std::vector<std::string> tensor_name_dest_list;
static std::vector<torch::Tensor> tensor_dest_list;
static bool dispatch_finished = false;
static bool dispatch_finished_dest = false;
static int dispatched_tensors = 0;
static int dispatched_tensors_dest = 0;
static json header;
static json header_dest;

static char *inBuffer = NULL;
static int64_t cur_recv_delta = 0;
static torch::Dict<std::string, torch::Tensor> state_dict;

static char *host_buffer = NULL;
const int64_t host_buffer_size = 2ll << 30;    //  2GB

static json get_header(std::string model_path) {
    std::ifstream infile(model_path);

    // get header size
    uint64_t header_size = 0;
    infile.read((char *) &header_size, 8);

    // get header
    char *header_buffer = (char *) malloc(header_size + 1);
    infile.read(header_buffer, header_size);
    header_buffer[header_size] = 0;

    return json::parse(header_buffer);
}

void do_dispatch() {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    /* Pre-allocate tensors */
    for (auto it = header.begin(); it != header.end(); it++) {
        std::string name = it.key();
        if (!name.compare("__metadata__")) continue;
        auto info = it.value();
        std::string dtype = info["dtype"].get<std::string>();
        std::vector<int64_t> shape = info["shape"].get<std::vector<int64_t>>();

        // auto options = torch::TensorOptions().dtype(getDtype(dtype)).device(torch::kCUDA, 0);
        auto options = torch::TensorOptions().dtype(getDtype(dtype)).device(torch::kCPU);
        auto tensor = torch::empty(at::IntArrayRef(shape), options);

        tensor_name_list.emplace_back(name);
        tensor_list.emplace_back(tensor);
    }

    /* Perform dispatch */
    int ptr = 0;
    for (auto it = header.begin(); it != header.end(); it++) {
        std::string name = it.key();
        if (!name.compare("__metadata__")) continue;
        auto info = it.value();
        int64_t start_pos = info["data_offsets"][0];
        int64_t end_pos = info["data_offsets"][1];

        while (end_pos > *cur_recv_end) usleep(1000);

        auto tensor = tensor_list[ptr++];
        void *pos = tensor.data_ptr();

        // CUDA_CHECK(cudaMemcpy(pos, myBuffer + start_pos, end_pos - start_pos, cudaMemcpyHostToDevice));
        memcpy(pos, myBuffer + start_pos, end_pos - start_pos);
        dispatched_tensors += 1;
    }
    std::atomic_thread_fence(std::memory_order_release);
    dispatch_finished = true;
    gettimeofday(&end, NULL);
    printf("dispatch model elapsed: %.3lf seconds\n", end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0f);
}

// downloading from the given safetensors path
bool start_load() {
    char *enable_para_str = getenv("ENABLE_PARA");
    int enable_para = enable_para_str ? atoi(enable_para_str) : 1;
    if (!enable_para) {
        // print to stderr to display in pod logs
        fprintf(stderr, "No parallel download.");
        fflush(stderr);
        return false;
    }

    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);

    std::string model_path = getenv("MODEL_PATH");

    header = get_header(model_path);

    // get data buffer size
    auto it = header.end(); it--;
    int64_t data_size = it.value()["data_offsets"][1];

    int shm_id = shmget(501, data_size + 8, 0644 | IPC_CREAT);
    myBuffer = (char *) shmat(shm_id, NULL, 0);
    cur_recv_end = (int64_t *) myBuffer;
    myBuffer += 8;

    std::thread dispatch_thread = std::thread(do_dispatch);
    dispatch_thread.detach();

    gettimeofday(&end, NULL);
    printf("start load model elapsed: %.3lf seconds\n", end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0f);

    return true;
}

// get state dict
static bool normal_returned = false;
static int last_dispatched_tensors = 0;
static std::tuple<std::string, torch::Tensor> empty_tuple("e", torch::empty(1));
static std::tuple<std::string, torch::Tensor> stop_tuple("s", torch::empty(1));
std::tuple<std::string, torch::Tensor> get_state_dict() {
    // if normal weights have not returned, check normal weights; else check dest model weights
    if (!normal_returned) {
        if (last_dispatched_tensors < dispatched_tensors) {
            auto ret = std::make_tuple(tensor_name_list[last_dispatched_tensors], tensor_list[last_dispatched_tensors]);
            last_dispatched_tensors++;
            return ret;
        }
        if (dispatch_finished) {
            // double-check dispatched_tensors to ensure that we have sent all tensors
            if (last_dispatched_tensors < dispatched_tensors) return empty_tuple;
            normal_returned = true;
            last_dispatched_tensors = 0;
            return stop_tuple;
        }
        return empty_tuple;
    }
    // for dest model
    if (last_dispatched_tensors < dispatched_tensors_dest) {
        auto ret = std::make_tuple(tensor_name_dest_list[last_dispatched_tensors], tensor_dest_list[last_dispatched_tensors]);
        last_dispatched_tensors++;
        return ret;
    }
    if (dispatch_finished_dest) return stop_tuple;
    return empty_tuple;
}

void delete_state_dict() {
    // delete all tensors in GPU
    // do nothing
}

// start download dest model
void load_dest_state_dict() {
    struct timeval start, end;

    std::string dest_model_path = getenv("DEST_MODEL_PATH");
    
    std::ifstream infile = std::ifstream(dest_model_path);

    // get header size
    uint64_t header_size = 0;
    infile.read((char *) &header_size, 8);

    // get header
    char *header_buffer = (char *) malloc(header_size + 1);
    infile.read(header_buffer, header_size);
    header_buffer[header_size] = 0;
    header_dest = json::parse(header_buffer);

    gettimeofday(&start, NULL);

    int64_t last_end_pos = 0;
    for (auto it = header_dest.begin(); it != header_dest.end(); it++) {
        std::string name = it.key();
        if (!name.compare("__metadata__")) continue;
        auto info = it.value();
        std::string dtype = info["dtype"].get<std::string>();
        std::vector<int64_t> shape = info["shape"].get<std::vector<int64_t>>();
        int64_t start_pos = info["data_offsets"][0];
        int64_t end_pos = info["data_offsets"][1];

        if (start_pos != last_end_pos) {
            printf("Dispatch Error: last_end_pos = %ld and start_pos = %ld which is not equal.\n", last_end_pos, start_pos);
        }

        // auto options = torch::TensorOptions().dtype(getDtype(dtype)).device(torch::kCUDA, 0);
        auto options = torch::TensorOptions().dtype(getDtype(dtype)).device(torch::kCPU);
        auto tensor = torch::empty(at::IntArrayRef(shape), options);
        char *pos = (char *) tensor.data_ptr();

        tensor_name_dest_list.emplace_back(name);
        tensor_dest_list.emplace_back(tensor);

        infile.read(pos, end_pos - start_pos);
        dispatched_tensors_dest += 1;
        last_end_pos = end_pos;
    }

    std::atomic_thread_fence(std::memory_order_release);
    dispatch_finished_dest = true;
    gettimeofday(&end, NULL);
    printf("dispatch dest model elapsed: %.3lf seconds\n", end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0f);
}

// allocate buffer for kv cache migration
void allocate_host_buffer() {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    cudaHostAlloc(&host_buffer, host_buffer_size, cudaHostAllocDefault);

    gettimeofday(&end, NULL);
    printf("allocate_host_buffer: alloc buffer for kv cache migration elapsed: %.3lf seconds\n", end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0f);
}

static json get_header_shm(char * & inBuffer, int64_t & cur_recv_delta, volatile int64_t *cur_recv_end) {
    // get header size
    printf("cur_recv_end = %ld\n", *cur_recv_end);
    while (*cur_recv_end < cur_recv_delta + 8)
        ;
    uint64_t header_size = *((uint64_t *) inBuffer);
    inBuffer += 8;
    cur_recv_delta += 8;

    // get header
    while (*cur_recv_end < cur_recv_delta + header_size)
        ;
    char *header_buffer = (char *) malloc(header_size + 1);
    memcpy(header_buffer, inBuffer, header_size);
    inBuffer += header_size;
    cur_recv_delta += header_size;
    header_buffer[header_size] = 0;

    return json::parse(header_buffer);
}

static bool normal_started = false;

void do_dispatch_shm(char *inBuffer) {
    struct timeval start, end, end_1;
    gettimeofday(&start, NULL);

    volatile int64_t *cur_recv_end = (int64_t *) inBuffer;
    inBuffer += 8;
    int64_t cur_recv_delta = 8;
    
    bool is_dest = true;
    if (!normal_started) {
        normal_started = true;
        is_dest = false;
    }

    /*
    [DEPRECATED] Do not pre-allocate buffer for kv cache migration because the allocation process impacts the inference performance
    if (is_dest) {
        std::thread allocate_host_buffer_thread = std::thread(allocate_host_buffer);
        allocate_host_buffer_thread.detach();
    }
    */

    header = get_header_shm(inBuffer, cur_recv_delta, cur_recv_end);

    /* Pre-allocate tensors */
    for (auto it = header.begin(); it != header.end(); it++) {
        std::string name = it.key();
        if (!name.compare("__metadata__")) continue;
        auto info = it.value();
        std::string dtype = info["dtype"].get<std::string>();
        std::vector<int64_t> shape = info["shape"].get<std::vector<int64_t>>();

        auto options = torch::TensorOptions().dtype(getDtype(dtype)).device(torch::kCUDA, 0).requires_grad(false);
        auto tensor = torch::empty(at::IntArrayRef(shape), options);

        if (is_dest) {
            tensor_name_dest_list.emplace_back(name);
            tensor_dest_list.emplace_back(tensor);
        } else {
            tensor_name_list.emplace_back(name);
            tensor_list.emplace_back(tensor);
        }
    }

    /* Create CUDA streams and assign tensors*/
    cudaStream_t stream;
    int leastPriority, greatestPriority;
    if (is_dest) {
        cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, leastPriority);
    }

    gettimeofday(&end, NULL);
    double allocate_time = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0f;

    /* Perform dispatch */
    int ptr = 0;
    for (auto it = header.begin(); it != header.end(); it++) {
        std::string name = it.key();
        if (!name.compare("__metadata__")) continue;
        auto info = it.value();
        int64_t start_pos = info["data_offsets"][0];
        int64_t end_pos = info["data_offsets"][1];

        while (*cur_recv_end < cur_recv_delta + end_pos)
            ;

        auto tensor = is_dest ? tensor_dest_list[ptr++] : tensor_list[ptr++];
        if (is_dest) {
            cudaMemcpyAsync(tensor.data_ptr(),
                            inBuffer + start_pos,
                            end_pos - start_pos,
                            cudaMemcpyHostToDevice,
                            stream);
        } else {
            cudaMemcpy(tensor.data_ptr(), inBuffer + start_pos, end_pos - start_pos, cudaMemcpyHostToDevice);
            dispatched_tensors++;
        }
    }
    if (is_dest) {
        cudaStreamSynchronize(stream);
        dispatched_tensors_dest = ptr;
    }

    std::atomic_thread_fence(std::memory_order_release);
    if (is_dest)
        dispatch_finished_dest = true;
    else
        dispatch_finished = true;

    gettimeofday(&end, NULL);
    if (is_dest) printf("Warning: For dest model, we load tensors in low-priority (%d) CUDA stream and return tensors only when all parameters have loaded.\n", leastPriority);
    printf("dispatch model: allocate tensors and stream time cost = %.3lf seconds\n", allocate_time);
    printf("dispatch model elapsed: %.3lf seconds\n", end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0f);
}

// downloading from the given safetensors path
void start_load_shm(std::string shm_name, int64_t shm_addr, int64_t shm_size) {
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);

    int shm_fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
    if (shm_fd == -1) {
        throw std::runtime_error("shm_open error.");
    }

    void *ptr = mmap(NULL, shm_size, PROT_READ, MAP_SHARED, shm_fd, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("mmap error.");
    }

    inBuffer = static_cast<char*>(ptr);
    inBuffer += shm_addr;

    std::thread dispatch_thread = std::thread(do_dispatch_shm, inBuffer);
    dispatch_thread.detach();

    gettimeofday(&end, NULL);
    printf("start load model elapsed: %.3lf seconds\n", end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0f);
}

int8_t check_load_fin_shm() {
    return dispatch_finished_dest;
}

std::tuple<std::string, torch::Tensor> get_state_dict_shm() {
    if (!normal_returned) {
        if (last_dispatched_tensors < dispatched_tensors) {
            auto ret = std::make_tuple(tensor_name_list[last_dispatched_tensors], tensor_list[last_dispatched_tensors]);
            last_dispatched_tensors++;
            return ret;
        }
        if (dispatch_finished) {
            // double-check dispatched_tensors to ensure that we have sent all tensors
            if (last_dispatched_tensors < dispatched_tensors) return empty_tuple;
            normal_returned = true;
            last_dispatched_tensors = 0;
            return stop_tuple;
        }
        return empty_tuple;
    }
    // for dest model
    if (last_dispatched_tensors < dispatched_tensors_dest) {
        auto ret = std::make_tuple(tensor_name_dest_list[last_dispatched_tensors], tensor_dest_list[last_dispatched_tensors]);
        last_dispatched_tensors++;
        return ret;
    }
    if (dispatch_finished_dest) return stop_tuple;
    return empty_tuple;
}

bool start_load_flag = start_load(); 

void send_tensors(std::vector<torch::Tensor> tensors, int64_t num_bytes_per_tensor, int64_t sock_fd) {
    struct timeval start;
    struct timeval end, end_1, end_2;
    gettimeofday(&start, NULL);

    int64_t num_tensors = tensors.size();
    int64_t tot_bytes = num_tensors * num_bytes_per_tensor;
    char *buffer = NULL;
    cudaHostAlloc(&buffer, tot_bytes, cudaHostAllocDefault);
    
    /* [DEPRECATED] Do not pre-allocate buffer for kv cache migration because the allocation process impacts the inference performance
    if (tot_bytes > host_buffer_size) {
        printf("Error: need %ld bytes for kv cache migration. Allocate new buffer.", tot_bytes);
        cudaHostAlloc(&buffer, tot_bytes, cudaHostAllocDefault);
    } else {
        while (!host_buffer)
            ;
        buffer = host_buffer;
    }
    */

    gettimeofday(&end, NULL);
    printf("send_tensors: allocate buffer of %ld bytes time cost = %.3lf seconds\n", tot_bytes, end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0f);

    const int num_stream = 16;
    cudaStream_t streams[num_stream];
    for (int i = 0; i < num_stream; i++) cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    
    gettimeofday(&end_1, NULL);
    printf("send_tensors: allocate stream time cost = %.3lf seconds\n", end_1.tv_sec - end.tv_sec + (end_1.tv_usec - end.tv_usec) / 1000000.0f);
    
    int num_tensors_per_stream = num_tensors / num_stream;
    int allocate_tensors[16];
    for (int i = 0; i < num_stream; i++) {
        allocate_tensors[i] = num_tensors_per_stream;
        if (i < num_tensors % num_stream) allocate_tensors[i]++;
        if (i > 0) allocate_tensors[i] += allocate_tensors[i-1];
    }
    int cur_stream = 0;
    for (int i = 0; i < num_tensors; i++) {
        if (allocate_tensors[cur_stream] == i) cur_stream++;
        cudaMemcpyAsync(buffer + i * num_bytes_per_tensor,
                        tensors[i].data_ptr(),
                        num_bytes_per_tensor,
                        cudaMemcpyDeviceToHost,
                        streams[cur_stream]);
    }
    for (int i = 0; i < num_stream; i++) {
        cudaStreamSynchronize(streams[i]);
        // send contents
        int64_t start_pos = !i ? 0 : allocate_tensors[i-1] * num_bytes_per_tensor;
        int64_t end_pos = allocate_tensors[i] * num_bytes_per_tensor;
        int64_t cur_send_len = start_pos;
        while (cur_send_len < end_pos) {
            int64_t new_send_len = send(sock_fd, buffer + cur_send_len, end_pos - cur_send_len, 0);
            cur_send_len += new_send_len;
        }
    }

    gettimeofday(&end_2, NULL);
    printf("send_tensors: copy and send tensors time cost = %.3lf seconds\n", end_2.tv_sec - end_1.tv_sec + (end_2.tv_usec - end_1.tv_usec) / 1000000.0f);
}

static bool copy_fin = false;

void recv_copy_tensors_(std::vector<torch::Tensor> origin_tensors, int64_t num_bytes_per_tensor, int64_t sock_fd) {
    struct timeval start;
    struct timeval end, end_1, end_2;
    gettimeofday(&start, NULL);

    int64_t num_tensors = origin_tensors.size();
    int64_t tot_bytes = num_tensors * num_bytes_per_tensor;
    char *buffer = NULL;
    cudaHostAlloc(&buffer, tot_bytes, cudaHostAllocDefault);

    /* [DEPRECATED] Do not pre-allocate buffer for kv cache migration because the allocation process impacts the inference performance
    if (tot_bytes > host_buffer_size) {
        printf("Error: need %ld bytes for kv cache migration. Allocate new buffer.", tot_bytes);
        cudaHostAlloc(&buffer, tot_bytes, cudaHostAllocDefault);
    } else {
        while (!host_buffer)
            ;
        buffer = host_buffer;
    }
    */

    gettimeofday(&end, NULL);
    printf("recv_copy_tensors: allocate buffer of %ld bytes time cost = %.3lf seconds\n", tot_bytes, end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0f);

    const int num_stream = 16;
    cudaStream_t streams[num_stream];
    int leastPriority, greatestPriority;
    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    for (int i = 0; i < num_stream; i++) {
        cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, leastPriority);
    }

    gettimeofday(&end_1, NULL);
    printf("recv_copy_tensors: allocate stream time cost = %.3lf seconds\n", end_1.tv_sec - end.tv_sec + (end_1.tv_usec - end.tv_usec) / 1000000.0f);

    int num_tensors_per_stream = num_tensors / num_stream;
    int allocate_tensors[16];
    for (int i = 0; i < num_stream; i++) {
        allocate_tensors[i] = num_tensors_per_stream;
        if (i < num_tensors % num_stream) allocate_tensors[i]++;
        if (i > 0) allocate_tensors[i] += allocate_tensors[i-1];
    }
    int cur_stream = -1;
    for (int i = 0; i < num_tensors; i++) {
        if (cur_stream == -1 || allocate_tensors[cur_stream] == i) {
            cur_stream++;
            // recv contents
            int64_t start_pos = !cur_stream ? 0 : allocate_tensors[cur_stream-1] * num_bytes_per_tensor;
            int64_t end_pos = allocate_tensors[cur_stream] * num_bytes_per_tensor;
            int64_t cur_recv_len = start_pos;
            while (cur_recv_len < end_pos) {
                int64_t new_recv_len = recv(sock_fd, buffer + cur_recv_len, end_pos - cur_recv_len, 0);
                cur_recv_len += new_recv_len;
            }
        }

        // copy contents
        cudaMemcpyAsync(origin_tensors[i].data_ptr(),
                        buffer + i * num_bytes_per_tensor,
                        num_bytes_per_tensor,
                        cudaMemcpyHostToDevice,
                        streams[cur_stream]);
    }

    for (int i = 0; i < num_stream; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    copy_fin = true;

    gettimeofday(&end_2, NULL);
    printf("recv_copy_tensors: recv and copy all tensors time cost = %.3lf seconds\n", end_2.tv_sec - end_1.tv_sec + (end_2.tv_usec - end_1.tv_usec) / 1000000.0f);
}

void recv_copy_tensors(std::vector<torch::Tensor> origin_tensors, int64_t num_bytes_per_tensor, int64_t sock_fd) {
    std::thread copy_thread = std::thread(recv_copy_tensors_, origin_tensors, num_bytes_per_tensor, sock_fd);
    copy_thread.detach();
}

int64_t check_copy_fin() {
    if (!copy_fin) return 0;
    copy_fin = false;
    return 1;
}

}