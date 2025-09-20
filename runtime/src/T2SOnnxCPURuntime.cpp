#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <core/session/onnxruntime_cxx_api.h>
#include <string>
#include <memory>
#include <vector>
#include <iostream>
#include <cstdint>

namespace py = pybind11;
constexpr int PREFILL_THREAD_NUM = 0; // 0 means using default thread number
constexpr int STEP_DECODE_THREAD_NUM = 1;
constexpr int HEAD_NUM = 24;
constexpr int KV_CACHE_PREPARED_LENGTH = 512;
#define CPP_PRINT(msg) py::print("[C++] " + std::string(msg))
using OrtValueShapeType = std::vector<int64_t>;

template <typename MajorType, typename KVType=MajorType>
class T2SOnnxCPURuntime {
public:
    T2SOnnxCPURuntime(std::string t2s_encoder_path, std::string t2s_first_stage_decoder_path, std::string t2s_stage_decoder_path) {
        reset(std::move(t2s_encoder_path), std::move(t2s_first_stage_decoder_path), std::move(t2s_stage_decoder_path));
    }
    void reset(std::string t2s_encoder_path, std::string t2s_first_stage_decoder_path, std::string t2s_stage_decoder_path) {
        k_cache_.clear();
        v_cache_.clear();
        y_emb_cache_.clear();
        y_.release();
        iteration_ = 0;
        kv_cache_seq_init_len_ = 0;
        y_init_len_ = 0;

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "T2SOnnxCPURuntime");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_options.SetIntraOpNumThreads(PREFILL_THREAD_NUM);
        session_options.SetInterOpNumThreads(PREFILL_THREAD_NUM);
        t2s_encoder_session_.reset(new Ort::Session(env, t2s_encoder_path.data(), session_options));
        t2s_first_stage_decoder_session_.reset(new Ort::Session(env, t2s_first_stage_decoder_path.data(), session_options));
        for(const auto& name : t2s_first_stage_decoder_session_->GetOutputNames()) {
            t2s_first_stage_decoder_output_names_.push_back(name);
        }

        session_options.SetIntraOpNumThreads(STEP_DECODE_THREAD_NUM);
        session_options.SetInterOpNumThreads(STEP_DECODE_THREAD_NUM);
        t2s_stage_decoder_session_.reset(new Ort::Session(env, t2s_stage_decoder_path.data(), session_options));
        for(const auto& name : t2s_stage_decoder_session_->GetInputNames()) {
            t2s_stage_decoder_input_names_.push_back(name);
        }
        for(const auto& name : t2s_stage_decoder_session_->GetOutputNames()) {
            t2s_stage_decoder_output_names_.push_back(name);
        }
    }
    //Accepting iput while checking memory continuity and shape
    void first_step_decode(py::array_t<int64_t, py::array::c_style | py::array::forcecast> ref_seq, 
                           py::array_t<int64_t, py::array::c_style | py::array::forcecast> text_seq, 
                           py::array_t<MajorType, py::array::c_style | py::array::forcecast> ref_bert, 
                           py::array_t<MajorType, py::array::c_style | py::array::forcecast> text_bert, 
                           py::array_t<MajorType, py::array::c_style | py::array::forcecast> ssl_content){
        py::buffer_info ref_seq_buffer = ref_seq.request();
        py::buffer_info text_seq_buffer = text_seq.request();
        py::buffer_info ref_bert_buffer = ref_bert.request();
        py::buffer_info text_bert_buffer = text_bert.request();
        py::buffer_info ssl_content_buffer = ssl_content.request();
        // Check shape
        if (ref_seq_buffer.ndim != 2 || ref_seq_buffer.shape[0] != 1)
            throw std::runtime_error("ref_seq must be 2-D with shape (1, ref_seq_len)!");
        if (text_seq_buffer.ndim != 2 || text_seq_buffer.shape[0] != 1)
            throw std::runtime_error("text_seq must be 2-D with shape (1, text_seq_len)!");
        if (ref_bert_buffer.ndim != 2 || ref_bert_buffer.shape[1] != 1024 || ref_bert_buffer.shape[0] != ref_seq_buffer.shape[1])
            throw std::runtime_error("ref_bert must be 2-D with shape (ref_seq_len, 1024)!");
        if (text_bert_buffer.ndim != 2 || text_bert_buffer.shape[1] != 1024 || text_bert_buffer.shape[0] != text_seq_buffer.shape[1])
            throw std::runtime_error("text_bert must be 2-D with shape (text_seq_len, 1024)!");
        if (ssl_content_buffer.ndim != 3 || ssl_content_buffer.shape[0] != 1 || ssl_content_buffer.shape[1] != 768)
            throw std::runtime_error("ssl_content must be 3-D with shape (1, 768, ssl_seq_len)!");
        
        iteration_ = 0;
        kv_cache_seq_init_len_ = ref_seq_buffer.shape[1] + text_seq_buffer.shape[1] + (ssl_content_buffer.shape[2]/2);
        y_init_len_ = ssl_content_buffer.shape[2] / 2; 
        int x_len = ref_seq_buffer.shape[1] + text_seq_buffer.shape[1];
        int prompts_len = y_init_len_;

        // CPU memory info
        Ort::MemoryInfo pre_allocated_memory_info("Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault);
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::IoBinding encoder_iobinding(*t2s_encoder_session_);

        // Prepare input shapes
        OrtValueShapeType ref_seq_shape({(int64_t)ref_seq_buffer.shape[0], (int64_t)ref_seq_buffer.shape[1]});
        OrtValueShapeType text_seq_shape({(int64_t)text_seq_buffer.shape[0], (int64_t)text_seq_buffer.shape[1]});
        OrtValueShapeType ref_bert_shape({(int64_t)ref_bert_buffer.shape[0], (int64_t)ref_bert_buffer.shape[1]});
        OrtValueShapeType text_bert_shape({(int64_t)text_bert_buffer.shape[0], (int64_t)text_bert_buffer.shape[1]});
        OrtValueShapeType ssl_content_shape({(int64_t)ssl_content_buffer.shape[0], (int64_t)ssl_content_buffer.shape[1], (int64_t)ssl_content_buffer.shape[2]});
        //zero copy input data to tensor
        auto ref_seq_tensor = Ort::Value::CreateTensor<int64_t>(pre_allocated_memory_info, (int64_t*)ref_seq_buffer.ptr, ref_seq_buffer.size, ref_seq_shape.data(), ref_seq_shape.size());
        auto text_seq_tensor = Ort::Value::CreateTensor<int64_t>(pre_allocated_memory_info, (int64_t*)text_seq_buffer.ptr, text_seq_buffer.size, text_seq_shape.data(), text_seq_shape.size());
        auto ref_bert_tensor = Ort::Value::CreateTensor<MajorType>(pre_allocated_memory_info, (MajorType*)ref_bert_buffer.ptr, ref_bert_buffer.size, ref_bert_shape.data(), ref_bert_shape.size());
        auto text_bert_tensor = Ort::Value::CreateTensor<MajorType>(pre_allocated_memory_info, (MajorType*)text_bert_buffer.ptr, text_bert_buffer.size, text_bert_shape.data(), text_bert_shape.size());
        auto ssl_content_tensor = Ort::Value::CreateTensor<MajorType>(pre_allocated_memory_info, (MajorType*)ssl_content_buffer.ptr, ssl_content_buffer.size, ssl_content_shape.data(), ssl_content_shape.size());
        
        encoder_iobinding.BindInput("ref_seq", ref_seq_tensor);
        encoder_iobinding.BindInput("text_seq", text_seq_tensor);
        encoder_iobinding.BindInput("ref_bert", ref_bert_tensor);
        encoder_iobinding.BindInput("text_bert", text_bert_tensor);
        encoder_iobinding.BindInput("ssl_content", ssl_content_tensor);

        OrtValueShapeType x_shape = {1, x_len, 512};
        Ort::Value x_tensor = Ort::Value::CreateTensor<MajorType>(allocator, x_shape.data(), x_shape.size());
        OrtValueShapeType prompts_shape = {1, prompts_len};
        Ort::Value prompts_tensor = Ort::Value::CreateTensor<int64_t>(allocator, prompts_shape.data(), prompts_shape.size());
        encoder_iobinding.BindOutput("x", x_tensor);
        encoder_iobinding.BindOutput("prompts", prompts_tensor);

        t2s_encoder_session_->Run(Ort::RunOptions{nullptr}, encoder_iobinding);
        encoder_iobinding.SynchronizeOutputs();

        //prepare kv_cache for first stage decoder
        k_cache_.clear();
        v_cache_.clear();
        y_emb_cache_.clear();
        y_emb_cache_.resize(1 * (y_init_len_ + KV_CACHE_PREPARED_LENGTH) * 512);
        for(int i = 0; i < HEAD_NUM; ++i){
            k_cache_.push_back(std::vector<KVType>((kv_cache_seq_init_len_ + KV_CACHE_PREPARED_LENGTH) * 1 * 512));
            v_cache_.push_back(std::vector<KVType>((kv_cache_seq_init_len_ + KV_CACHE_PREPARED_LENGTH) * 1 * 512));
        }

        Ort::IoBinding first_step_decoder_iobinding(*t2s_first_stage_decoder_session_);
        // Bind inputs
        first_step_decoder_iobinding.BindInput("x", x_tensor);
        first_step_decoder_iobinding.BindInput("prompts", prompts_tensor);

        
        // Prepare and bind y and y_emb outputs
        OrtValueShapeType y_shape = {1, y_init_len_+ 1};
        y_ = Ort::Value::CreateTensor<int64_t>(allocator, y_shape.data(), y_shape.size());
        OrtValueShapeType y_emb_shape = {1, y_init_len_, 512};
        int y_emb_size = 1 * y_init_len_ * 512;
        auto y_emb = Ort::Value::CreateTensor<MajorType>(pre_allocated_memory_info, y_emb_cache_.data(), y_emb_size, y_emb_shape.data(), y_emb_shape.size());
        first_step_decoder_iobinding.BindOutput("y", y_);
        first_step_decoder_iobinding.BindOutput("y_emb", y_emb);
        // Prepare and bind kv_cache inputs
        OrtValueShapeType kv_cache_shape = {kv_cache_seq_init_len_, 1, 512};
        int kv_cache_init_size = kv_cache_seq_init_len_ * 1 * 512;
        for(int i = 0; i < HEAD_NUM; ++i){
            auto k_tensor = Ort::Value::CreateTensor<KVType>(pre_allocated_memory_info, k_cache_[i].data(), kv_cache_init_size, kv_cache_shape.data(), kv_cache_shape.size());
            auto v_tensor = Ort::Value::CreateTensor<KVType>(pre_allocated_memory_info, v_cache_[i].data(), kv_cache_init_size, kv_cache_shape.data(), kv_cache_shape.size());
            first_step_decoder_iobinding.BindOutput(t2s_first_stage_decoder_output_names_[2 + i * 2].c_str(), k_tensor);
            first_step_decoder_iobinding.BindOutput(t2s_first_stage_decoder_output_names_[2 + i * 2 + 1].c_str(), v_tensor);
        }

        t2s_first_stage_decoder_session_->Run(Ort::RunOptions{}, first_step_decoder_iobinding);
        first_step_decoder_iobinding.SynchronizeOutputs();
        auto first_step_decoder_outputs = first_step_decoder_iobinding.GetOutputValues();
    }

    bool step_decode(){
        assert(k_cache_.size() == HEAD_NUM && v_cache_.size() == HEAD_NUM);
        assert(y_.IsTensor());
        assert(t2s_stage_decoder_session_ != nullptr);
        int current_y_len = y_init_len_ + iteration_ + 1;
        int current_y_emb_len = y_init_len_ + iteration_;
        int current_kv_cache_len = kv_cache_seq_init_len_ + iteration_;
        int kv_cache_current_size = current_kv_cache_len * 1 * 512;
        int y_emb_current_size = 1 * current_y_emb_len * 512;

         // CPU memory info
        Ort::MemoryInfo pre_allocated_memory_info("Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault);
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::IoBinding stage_decoder_iobinding(*t2s_stage_decoder_session_);

        // Bind inputs
        stage_decoder_iobinding.BindInput("iy", y_);
        OrtValueShapeType y_emb_shape = {1, current_y_emb_len, 512};
        auto y_emb = Ort::Value::CreateTensor<MajorType>(pre_allocated_memory_info, y_emb_cache_.data(), y_emb_current_size, y_emb_shape.data(), y_emb_shape.size());
        stage_decoder_iobinding.BindInput("iy_emb", y_emb);
        OrtValueShapeType kv_cache_shape = {current_kv_cache_len, 1, 512};
        for(int i = 0; i < HEAD_NUM; ++i){
            auto k_tensor = Ort::Value::CreateTensor<KVType>(pre_allocated_memory_info, k_cache_[i].data(), kv_cache_current_size, kv_cache_shape.data(), kv_cache_shape.size());
            auto v_tensor = Ort::Value::CreateTensor<KVType>(pre_allocated_memory_info, v_cache_[i].data(), kv_cache_current_size, kv_cache_shape.data(), kv_cache_shape.size());
            stage_decoder_iobinding.BindInput(t2s_stage_decoder_input_names_[2 + i * 2].c_str(), k_tensor);
            stage_decoder_iobinding.BindInput(t2s_stage_decoder_input_names_[2 + i * 2 + 1].c_str(), v_tensor);
        }

        // Prepare and bind outputs
        OrtValueShapeType y_new_shape = {1, current_y_len + 1};
        Ort::Value y_new = Ort::Value::CreateTensor<int64_t>(allocator, y_new_shape.data(), y_new_shape.size());
        stage_decoder_iobinding.BindOutput("y", y_new);
        OrtValueShapeType y_emb_new_shape = {1, 1, 512};
        int y_emb_increased_size = 1 * 1 * 512;
        MajorType* y_emb_out_ptr = y_emb_cache_.data() + y_emb_current_size;
        Ort::Value y_emb_new = Ort::Value::CreateTensor<MajorType>(pre_allocated_memory_info, y_emb_out_ptr, y_emb_increased_size, y_emb_new_shape.data(), y_emb_new_shape.size());
        stage_decoder_iobinding.BindOutput("increased_y_emb", y_emb_new);
        bool stop_condition_data = false;
        Ort::Value stop_condition_tensor = Ort::Value::CreateTensor<bool>(pre_allocated_memory_info, &stop_condition_data, 1, nullptr, 0);
        stage_decoder_iobinding.BindOutput("stop_condition_tensor", stop_condition_tensor);


        OrtValueShapeType kv_cache_shape_out = {1, 1, 512};
        int kv_cache_increased_size = 1 * 1 * 512;
        for(int i = 0; i < HEAD_NUM; ++i){
            KVType* k_cache_out_ptr = k_cache_[i].data() + kv_cache_current_size;
            KVType* v_cache_out_ptr = v_cache_[i].data() + kv_cache_current_size;
            auto k_tensor = Ort::Value::CreateTensor<KVType>(pre_allocated_memory_info, k_cache_out_ptr, kv_cache_increased_size, kv_cache_shape_out.data(), kv_cache_shape_out.size());
            auto v_tensor = Ort::Value::CreateTensor<KVType>(pre_allocated_memory_info, v_cache_out_ptr, kv_cache_increased_size, kv_cache_shape_out.data(), kv_cache_shape_out.size());
            stage_decoder_iobinding.BindOutput(t2s_stage_decoder_output_names_[3 + i * 2].c_str(), k_tensor);
            stage_decoder_iobinding.BindOutput(t2s_stage_decoder_output_names_[3 + i * 2 + 1].c_str(), v_tensor);
        }

        // CPP_PRINT("Running stage decoder...");
        t2s_stage_decoder_session_->Run(Ort::RunOptions{}, stage_decoder_iobinding);
        // CPP_PRINT("Stage decoder run completed.");
        stage_decoder_iobinding.SynchronizeOutputs();

        std::swap(y_, y_new);
        iteration_ += 1;

        return stop_condition_data;
    }

    py::array_t<int64_t> get_current_y() {
        assert(y_.IsTensor());
        int64_t* y_data = y_.GetTensorMutableData<int64_t>();
        auto y_array = py::array_t<int64_t>(y_.GetTensorTypeAndShapeInfo().GetShape());
        py::buffer_info buf = y_array.request();
        std::memcpy(buf.ptr, y_data, sizeof(int64_t) * buf.size);
        return y_array;
    }

    py::array_t<int64_t> run(py::array_t<int64_t, py::array::c_style | py::array::forcecast> ref_seq, 
                                py::array_t<int64_t, py::array::c_style | py::array::forcecast> text_seq, 
                                py::array_t<MajorType, py::array::c_style | py::array::forcecast> ref_bert, 
                                py::array_t<MajorType, py::array::c_style | py::array::forcecast> text_bert, 
                                py::array_t<MajorType, py::array::c_style | py::array::forcecast> ssl_content){
        first_step_decode(ref_seq, text_seq, ref_bert, text_bert, ssl_content);
        int i;
        for(i = 0; i < KV_CACHE_PREPARED_LENGTH; ++i){
            bool stop = step_decode();
            if(stop) break;
        }
        int y_num = y_.GetTensorTypeAndShapeInfo().GetElementCount();
        int64_t* y_data = y_.GetTensorMutableData<int64_t>();
        // Getting y[-i-1:-1]
        int64_t* semantics_start = y_data + (y_num - (i + 1));
        auto semantics_array = py::array_t<int64_t>({1, 1, i - 1});
        py::buffer_info buf = semantics_array.request();
        std::memcpy(buf.ptr, semantics_start, sizeof(int64_t) * (i - 1));
        return semantics_array;
    }
private:
    std::vector<std::vector<KVType>> k_cache_;
    std::vector<std::vector<KVType>> v_cache_;
    std::vector<MajorType> y_emb_cache_;
    Ort::Value y_;
    std::unique_ptr<Ort::Session> t2s_encoder_session_;
    std::unique_ptr<Ort::Session> t2s_first_stage_decoder_session_;
    std::unique_ptr<Ort::Session> t2s_stage_decoder_session_;
    int iteration_ = 0;
    int kv_cache_seq_init_len_ = 0;
    int y_init_len_ = 0;
    std::vector<std::string> t2s_first_stage_decoder_output_names_;
    std::vector<std::string> t2s_stage_decoder_input_names_;
    std::vector<std::string> t2s_stage_decoder_output_names_;
};

using T2SOnnxCPURuntimeF32 = T2SOnnxCPURuntime<float, float>;
using T2SOnnxCPURuntimeF16 = T2SOnnxCPURuntime<uint16_t, uint16_t>; // Using uint16_t to represent float16

#define BIND_TEMPLATE_CLASS(className, pyName, major_type, kv_type) \
    py::class_<className<major_type, kv_type>>(m, pyName) \
        .def(py::init<std::string, std::string, std::string>()) \
        .def("first_step_decode", &className<major_type, kv_type>::first_step_decode) \
        .def("step_decode", &className<major_type, kv_type>::step_decode) \
        .def("get_current_y", &className<major_type, kv_type>::get_current_y) \
        .def("run", &className<major_type, kv_type>::run) \
        .def("reset", &className<major_type, kv_type>::reset)


PYBIND11_MODULE(T2SOnnxCPURuntime, m) {
    m.doc() = "T2SOnnxCPURuntime Implementations"; 
    BIND_TEMPLATE_CLASS(T2SOnnxCPURuntime, "T2SOnnxCPURuntimeF32", float, float);
    BIND_TEMPLATE_CLASS(T2SOnnxCPURuntime, "T2SOnnxCPURuntimeF16", uint16_t, uint16_t);
}