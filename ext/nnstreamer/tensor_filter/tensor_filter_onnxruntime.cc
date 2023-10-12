#include <iostream>
#include <sstream>

#include <glib.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_util.h>
#include <tensor_common.h>

#include <core/session/onnxruntime_cxx_api.h>


/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif


namespace nnstreamer
{
namespace tensor_filter_onnxruntime
{
extern "C" {
void init_filter_onnxruntime (void) __attribute__ ((constructor));
void fini_filter_onnxruntime (void) __attribute__ ((destructor));
}

/** @brief tensor-filter-subplugin concrete class for onnxruntime */
class onnxruntime_subplugin final : public tensor_filter_subplugin
{
  private:
  bool empty_model;
  char *model_path; /**< The model *.onnx file */

  GstTensorsInfo inputInfo; /**< Input tensors metadata */
  GstTensorsInfo outputInfo; /**< Output tensors metadata */

  Ort::Session session;
  Ort::MemoryInfo memInfo;

  // tensor setting
  std::size_t input_num_tensors;
  std::vector<const char*> input_names;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<ONNXTensorElementDataType> input_types;
  std::vector<Ort::AllocatedStringPtr> input_names_allocated_strings;
  std::vector<Ort::Value> input_tensors; /**< Input tensor from model */

  std::size_t output_num_tensors;
  std::vector<const char*> output_names;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<ONNXTensorElementDataType> output_types;
  std::vector<Ort::AllocatedStringPtr> output_names_allocated_strings;
  std::vector<Ort::Value> output_tensors; /**< Output tensor from model */

  static const char *name; 
  static onnxruntime_subplugin *registeredRepresentation;

  void cleanup();
  std::string print_shape(const std::vector<std::int64_t>& v);
  int getTensorDim (tensor_dim &dim, std::vector<int64_t> shapes, size_t numDims);
  int getTensorType (ONNXTensorElementDataType _type, tensor_type *type);

  public:
  static void init_filter_onnxruntime ();
  static void fini_filter_onnxruntime ();

  onnxruntime_subplugin ();
  ~onnxruntime_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

const char *onnxruntime_subplugin::name = "onnxruntime";

/**
 * @brief Constructor for onnxruntime_subplugin.
 */
onnxruntime_subplugin::onnxruntime_subplugin ()
    : tensor_filter_subplugin (), empty_model (true), model_path (nullptr), 
    session (nullptr), memInfo (nullptr), 
    input_num_tensors (0U), input_names (), input_shapes (), input_types (), input_names_allocated_strings (), input_tensors (),
    output_num_tensors (0U), output_names (), output_shapes (), output_types (), output_names_allocated_strings (), output_tensors ()
{
  gst_tensors_info_init (std::addressof (inputInfo));
  gst_tensors_info_init (std::addressof (outputInfo));

  input_tensors.reserve(NNS_TENSOR_SIZE_LIMIT);
  output_tensors.reserve(NNS_TENSOR_SIZE_LIMIT);

}

/** 
 * @brief Destructor for onnxruntime_subplugin. 
 */
onnxruntime_subplugin::~onnxruntime_subplugin ()
{
  gst_tensors_info_free (std::addressof (inputInfo));
  gst_tensors_info_free (std::addressof (outputInfo));
}

/** @brief cleanup resources used by onnxruntime subplugin */
void
onnxruntime_subplugin::cleanup ()
{
  if (empty_model)
    return; /* Nothing to do if it is an empty model */

  if (session) {
    session = Ort::Session{nullptr}; /* it's already freed with session */
  }

  if(memInfo) {
    memInfo = Ort::MemoryInfo{nullptr}; /* it's already freed with meminfo */
  }

  gst_tensors_info_free (std::addressof (inputInfo));
  gst_tensors_info_free (std::addressof (outputInfo));

  input_num_tensors = 0;
  input_names.clear();
  input_shapes.clear();
  input_types.clear();
  input_names_allocated_strings.clear();
  input_tensors.clear();

  output_num_tensors = 0;
  output_names.clear();
  output_shapes.clear();
  output_types.clear();
  output_names_allocated_strings.clear();
  output_tensors.clear();
  
  g_free (model_path);
  model_path = nullptr;
  empty_model = true;
}

/**
 * @brief pretty prints a shape dimension vector
 * @return the shape of tensor. for example "1x3x224x224".
 */
std::string
onnxruntime_subplugin::print_shape(const std::vector<std::int64_t>& v) {
  std::stringstream ss("");
  for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

/**
 * @brief return the shape of tensor
 * @return 0 if OK. non-zero if error.
 */
int
onnxruntime_subplugin::getTensorDim (tensor_dim &dim, std::vector<int64_t> shapes, size_t dims)
{
  size_t i;

  if (dims > NNS_TENSOR_RANK_LIMIT) {
    nns_logw ("Shape rank too high: %zu max: %d", dims, NNS_TENSOR_RANK_LIMIT);
    return -EINVAL;
  }

  /* the order of dimension is reversed at CAPS negotiation */
  for (i = 0; i < dims; i++)
    dim[i] = shapes[dims - i - 1]; 

  /* fill remaining entries with 0 */
  for (i = dims; i < NNS_TENSOR_RANK_LIMIT; ++i) {
    dim[i] = 0;
  }

  return 0;
}

/**
 * @brief return the type of tensor
 * @return 0 if OK. non-zero if error.
 */
int
onnxruntime_subplugin::getTensorType (ONNXTensorElementDataType _type, tensor_type *type)
{
  tensor_type res;
  *type = _NNS_END;

  switch (_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      res = _NNS_INT8;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      res = _NNS_UINT8;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      res = _NNS_INT16;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      res = _NNS_UINT16;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      res = _NNS_INT32;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      res = _NNS_UINT32;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      res = _NNS_INT64;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      res = _NNS_UINT64;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      res = _NNS_FLOAT32;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      res = _NNS_FLOAT64;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      res = _NNS_FLOAT16;
      break;
    default:
      nns_logw ("Tensor type not supported: %d", (gint) _type);
      return -EINVAL;
  }

  *type = res;
  return 0;

}

/**
 * @brief Method to get empty object.
 */
tensor_filter_subplugin &
onnxruntime_subplugin::getEmptyInstance ()
{
  return *(new onnxruntime_subplugin ());
}

/**
 * @brief Method to prepare/configure onnxruntime instance.
 */
void
onnxruntime_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  size_t i;
  Ort::SessionOptions sessionOptions;

  if (!empty_model) {
    /* Already opened */

    if (!prop->model_files[0] || prop->model_files[0][0] == '\0') {
      std::cerr << "Model path is not given." << std::endl;
      throw std::invalid_argument ("Model path is not given.");
    }
    
    cleanup ();
  }

  assert (model_path == nullptr);

  if (!g_file_test (prop->model_files[0], G_FILE_TEST_IS_REGULAR)) {
    const std::string err_msg
        = "Given file " + (std::string) prop->model_files[0] + " is not valid";
    std::cerr << err_msg << std::endl;
    cleanup ();
    throw std::invalid_argument (err_msg);
  }

  model_path = g_strdup (prop->model_files[0]);

  assert (model_path != nullptr);

  if (!g_file_test (prop->model_files[0], G_FILE_TEST_IS_REGULAR)) {
    const std::string err_msg
        = "Given file " + (std::string) prop->model_files[0] + " is not valid";
    std::cerr << err_msg << std::endl;
    cleanup ();
    throw std::invalid_argument (err_msg);
  }

  /** Read a model */
  Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "nnstreamer_onnxruntime");
  Ort::AllocatorWithDefaultOptions allocator;
  session = Ort::Session(env, model_path, sessionOptions);
  
  input_num_tensors = session.GetInputCount();
  if (input_num_tensors <= 0 || input_num_tensors > NNS_TENSOR_SIZE_LIMIT){
    cleanup ();
    throw std::invalid_argument (
        "The number of tensors required by the given model exceeds the nnstreamer tensor limit (" NNS_TENSOR_SIZE_LIMIT_STR
        " by default).");

  }
  inputInfo.num_tensors = input_num_tensors;

  output_num_tensors = session.GetOutputCount();
  if (input_num_tensors <= 0 || input_num_tensors > NNS_TENSOR_SIZE_LIMIT) {
    cleanup ();
    throw std::invalid_argument (
        "The number of tensors required by the given model exceeds the nnstreamer tensor limit (" NNS_TENSOR_SIZE_LIMIT_STR
        " by default).");
  }
  outputInfo.num_tensors = output_num_tensors;
  

  // initialize input info
  input_shapes.resize(inputInfo.num_tensors);
  input_types.resize(inputInfo.num_tensors);

  for (i = 0; i < inputInfo.num_tensors; i++){

    // Get input names
    auto input_name = session.GetInputNameAllocated(i, allocator);
    input_names_allocated_strings.push_back(std::move(input_name));
    input_names.push_back(input_names_allocated_strings.back().get());

    // Get input types
    ONNXTensorElementDataType type = 
                      session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();

    if (getTensorType (type, &inputInfo.info[i].type)) {
      cleanup();
      throw std::invalid_argument ("Failed to convert ONNXTensorElement intput data type");
    }
    input_types[i] = type;

    // Get input shapes/dims
    size_t num_dims = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetDimensionsCount();
    input_shapes[i] =  session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

    // free dimensions are treated as 1 if not overriden
    for (int64_t& dim : input_shapes[i]) {
      if (dim == -1) {
        dim = 1;
      }
    }

    std::cout << "\t" << input_names.at(i) << " : " << print_shape(input_shapes[i]) << std::endl;

    if (getTensorDim (inputInfo.info[i].dimension, input_shapes[i], num_dims)){
      cleanup();
      throw std::invalid_argument ("Shape input rank too high.");
    }

    inputInfo.info[i].name = nullptr;

    gchar *dim;
    dim = gst_tensor_get_dimension_string (inputInfo.info[i].dimension);

    nns_logd ("inputInfo[%zu] >> name[%s], type[%d], dim[%s]", i,
        inputInfo.info[i].name, inputInfo.info[i].type, dim);
    g_free (dim);
  }
  
  // initialize output info
  output_shapes.resize(outputInfo.num_tensors);
  output_types.resize(outputInfo.num_tensors);

  for (i = 0; i < outputInfo.num_tensors; i++){

    // Get output node names
    auto output_name = session.GetOutputNameAllocated(i, allocator);
    output_names_allocated_strings.push_back(std::move(output_name));
    output_names.push_back(output_names_allocated_strings.back().get());

    // Get output node types
    ONNXTensorElementDataType type = 
                      session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();

    if (getTensorType (type, &outputInfo.info[i].type)) {
      cleanup();
      throw std::invalid_argument ("Failed to convert ONNXTensorElement output data type");
    }
    output_types[i] = type;

    // Get output shapes/dims
    size_t num_dims = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetDimensionsCount();
    output_shapes[i] =  session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

    // free dimensions are treated as 1 if not overriden
    for (int64_t& dim : output_shapes[i]) {
      if (dim == -1) {
        dim = 1;
      }
    }

    std::cout << "\t" << output_names.at(i) << " : " << print_shape(output_shapes[i]) << std::endl;

    if (getTensorDim (outputInfo.info[i].dimension, output_shapes[i], num_dims)){
      cleanup();
      throw std::invalid_argument ("Shape output rank too high.");
    }
    outputInfo.info[i].name = nullptr;

    gchar *dim;
    dim = gst_tensor_get_dimension_string (outputInfo.info[i].dimension);

    nns_logd ("outputInfo[%zu] >> name[%s], type[%d], dim[%s]", i,
        outputInfo.info[i].name, outputInfo.info[i].type, dim);
    g_free (dim);
  }

  empty_model = false;
  sessionOptions = Ort::SessionOptions{ nullptr };
  allocator = Ort::AllocatorWithDefaultOptions{ nullptr }; /* delete unique_ptr */

  printf ("onnxruntime configure instance setting \n");
}

/**
 * @brief Method to execute the model.
 */
void
onnxruntime_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  size_t i;
  assert (!empty_model);
  input_tensors.clear();
  output_tensors.clear();

  memInfo =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  assert (input_shapes.size() == inputInfo.num_tensors);

  if (!input)
    throw std::runtime_error ("Invalid input buffer, it is NULL.");
  if (!output)
    throw std::runtime_error ("Invalid output buffer, it is NULL.");

  // Set input to tensor
  for (i = 0; i < inputInfo.num_tensors; ++i) {
    input_tensors.emplace_back(Ort::Value::CreateTensor(memInfo, input[i].data, 
                                input[i].size, input_shapes[i].data(),
                                input_shapes[i].size(), input_types[i]));
  }

  // double-check the dimensions of the input tensor
  assert(input_tensors[0].IsTensor() && input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() == input_shapes[0]);
  
  // Set output to tensor
  for(i = 0; i < outputInfo.num_tensors; ++i){
    output_tensors.emplace_back(Ort::Value::CreateTensor(memInfo, output[i].data, output[i].size,
                                                     output_shapes[i].data(), output_shapes[i].size(), output_types[i]));
  }

  printf ("Running model... \n");
  try {
    // call Run() to fill in the GstTensorMemory *output data with the probabilities of each
    session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_names.size(), 
                output_names.data(), output_tensors.data(), output_names.size());

    printf ("Done! \n");

    // double-check the dimensions of the output tensor
    assert(output_tensors.size() == output_names.size() && output_tensors[0].IsTensor());

  } catch (const Ort::Exception& exception) {
      std::cerr <<  "ERROR running model inference: " << exception.what() << std::endl;
      cleanup ();
      throw std::invalid_argument ("exception.");
  }
  printf ("invoke \n");
}

/**
 * @brief Method to get the information of onnxruntime subplugin.
 */
void
onnxruntime_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = name;
  info.allow_in_place = 0;
  info.allocate_in_invoke = 0;
  info.run_without_model = 0;
  info.verify_model_path = 1;
}

/**
 * @brief Method to get the model information.
 */
int
onnxruntime_subplugin::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  if (ops == GET_IN_OUT_INFO) {
    gst_tensors_info_copy (std::addressof (in_info), std::addressof (inputInfo));
    gst_tensors_info_copy (std::addressof (out_info), std::addressof (outputInfo));
    return 0;
  }

  return -ENOENT;
}

/**
 * @brief Method to handle events.
 */
int
onnxruntime_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  UNUSED (ops);
  UNUSED (data);
  return -ENOENT;
}

onnxruntime_subplugin *onnxruntime_subplugin::registeredRepresentation = nullptr;


void
onnxruntime_subplugin::init_filter_onnxruntime ()
{
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<onnxruntime_subplugin> ();

      
}

/**
 * @brief Destruct the sub-plugin for onnxruntime.
 */
void
onnxruntime_subplugin::fini_filter_onnxruntime ()
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/** @brief initializer */
void
init_filter_onnxruntime ()
{
  onnxruntime_subplugin::init_filter_onnxruntime ();
    if (nnstreamer_filter_find ("onnx")) {
    nns_loge ("Cannot use onnxruntime and onnx both. Won't register this onnxruntime subplugin.");
    return;
  }
}

/** @brief finalizer */
void
fini_filter_onnxruntime ()
{
  onnxruntime_subplugin::fini_filter_onnxruntime ();
}

} // namespace tensor_filter_onnxruntime
} /* namespace nnstreamer */
