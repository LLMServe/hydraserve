// download the model weights
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
#include <errno.h>
#include <unistd.h>
#include <sys/time.h>
#include <unordered_map>
#include <fstream>
#include <sys/time.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <thread>
#include <omp.h>
#include "json.hpp"
using namespace std;
using json = nlohmann::json;

#define CHUNK_SIZE (1 << 25)    // 32MB

void download_model(string model_path, int shm_key) {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    std::ifstream infile = std::ifstream(model_path);

    // get header size
    uint64_t header_size = 0;
    infile.read((char *) &header_size, 8);

    // get header
    char *header_buffer = (char *) malloc(header_size + 1);
    infile.read(header_buffer, header_size);
    header_buffer[header_size] = 0;
    json header = json::parse(header_buffer);

    // get data buffer size
    auto it = header.end(); it--;
    int64_t data_size = it.value()["data_offsets"][1];
    printf("data size = %ld bytes\n", data_size);

    // create shared memory region | cur_recv_end (8 bytes) | data buffer (#data_size bytes) |
    int shm_id = shmget(shm_key, data_size + 8, 0644 | IPC_CREAT);     // TODO: whether to use huge page?
    char *myBuffer = (char *) shmat(shm_id, NULL, 0);
    // TODO: whether to pre-read all pages?
    // for (int64_t i = 0; i < data_size + 8; i += 4096)
    //     myBuffer[i] = 0;
    volatile int64_t *cur_recv_end = (int64_t *) myBuffer;
    *cur_recv_end = 0;
    myBuffer += 8;

    gettimeofday(&end, NULL);
    printf("download_model: initialization time cost = %.3lf seconds.\n", end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0f);

    const int parallel_size = 4;
    char *start_pos[parallel_size];
    int64_t portion = data_size / parallel_size;
    for (int i = 0; i < parallel_size; i++) {
        start_pos[i] = myBuffer + portion * i;
    }

    #pragma omp parallel num_threads(parallel_size) \
    shared(model_path, start_pos, header_size, data_size, cur_recv_end)
    {
        FILE *f = fopen(model_path.c_str(), "r");
        const unsigned int idx = omp_get_thread_num();
        const unsigned int tot = omp_get_num_threads();
        int64_t portion = data_size / tot;
        const int64_t start = portion * idx + header_size + 8;
        if (idx == tot - 1) portion += data_size % tot;
        fseek(f, start, SEEK_SET);
        unsigned int n;
        char *pos = start_pos[idx];
        size_t cur_read_bytes = 0;
        while (cur_read_bytes < portion) {
            size_t new_bytes = fread(pos, 1, portion - cur_read_bytes, f);
            cur_read_bytes += new_bytes;
            pos += new_bytes;
            if (!idx) {
                *cur_recv_end = cur_read_bytes;
            }
        }
        fclose(f);
    }

    *cur_recv_end = data_size;
}

void download(string model_path) {
    struct timeval start, end, end_1;
    gettimeofday(&start, NULL);

    download_model(model_path, 501);

    gettimeofday(&end, NULL);
    printf("total download model elapsed: %.3lf seconds\n", end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0f);
}

string get_model_path() {
    // Get path to model weights
    string FC_MODEL_CACHE_DIR = getenv("MODELSCOPE_CACHE");
    int pp_rank = atoi(getenv("PP_RANK"));
    int pp_size = atoi(getenv("PP_SIZE"));
    string model_id = getenv("MODEL_ID");
    // replace all "." to "___"
    size_t pos = model_id.find("."); 
    while (pos != string::npos) {
        model_id.replace(pos, 1, "___");
        pos = model_id.find(".", pos + 1);
    }
    string pp_tail = "";
    if (pp_size > 1) pp_tail = '-' + to_string(pp_rank) + '-' + to_string(pp_size);

    string model_path = FC_MODEL_CACHE_DIR + '/' + model_id + pp_tail + "/model.safetensors";

    // use vmtouch to evict page cache
    string command = "vmtouch -e " + FC_MODEL_CACHE_DIR + '/' + model_id + pp_tail;
    system(command.c_str());

    return model_path;
}

int main() {
    char *enable_para_str = getenv("ENABLE_PARA");
    int enable_para = enable_para_str ? atoi(enable_para_str) : 1;
    // print to stderr to display in pod logs
    fprintf(stderr, "Enable Parallel Downloading: %d\n", enable_para);
    fflush(stderr);
    string model_path = get_model_path();
    if (enable_para == 0) {
        system("python3 -u /vllm-workspace/app.py");
        // never return

        throw runtime_error("python program exited unexpectedly.");
    } else {
        int pid = fork();
        if (pid > 0) {
            // parent process

            system("python3 -u /vllm-workspace/app.py");
            // never return

            throw runtime_error("python program exited unexpectedly.");
        } else {
            // child process

            download(model_path);

            // sleep to make sure that the shared memory region exists
            fflush(stdout);
            while (true) {
                sleep(1000);
            }
        }
    }
    throw runtime_error("docker entry program exited unexpectedly.");
    return 1;
}