#include "common.h"

TORCH_LIBRARY(download_model, m) {
  m.def("get_state_dict", &st::get_state_dict);
  m.def("load_dest_state_dict", &st::load_dest_state_dict);
  m.def("delete_state_dict", &st::delete_state_dict);
  m.def("get_state_dict_shm", &st::get_state_dict_shm);
  m.def("start_load_shm", &st::start_load_shm);
  m.def("check_load_fin_shm", &st::check_load_fin_shm);
  m.def("send_tensors", &st::send_tensors);
  m.def("recv_copy_tensors", &st::recv_copy_tensors);
  m.def("check_copy_fin", &st::check_copy_fin);
}