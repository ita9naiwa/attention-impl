#include <torch/extension.h>

#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_attention", &naive_attention, "naive attention forward");
    m.def("single_query_attention", &single_query_attention, "kv forward");
    m.def("packed_attention", &packed_attention, "naive attention forward");
    m.def("kv_single_query_attention", &kv_single_query_attention, "kv forward");
    m.def("kv_multi_query_attention", &kv_multi_query_attention, "kv multi-query forward");
    m.def("rotary_embedding_inplace", &rotary_embedding_inplace, "rotary_embedding_inplace");
    m.def("layernorm_inplace", &layernorm_inplace, "layernorm_inplace");
};
