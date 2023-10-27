#include <gtest/gtest.h>
#include <glib.h>
#include <glib/gstdio.h> /* GStatBuf */
#include <gst/gst.h>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_util.h>
#include <tensor_common.h>
#include <unittest_util.h>
#include "nnstreamer_plugin_api.h"
#include "nnstreamer_plugin_api_util.h"

#define GTEST_COUT std::cerr << "[   INFO   ] "

/**
 * @brief internal function to get model filename
 */
static gboolean
_GetModelFilePath (gchar **model_file)
{
  const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();
  std::string model_name = "mobilenet_v2_float.onnx";

  *model_file = g_build_filename (
      root_path, "tests", "test_models", "models", model_name.c_str (), NULL);

  GTEST_COUT << "model path = " << *model_file << std::endl;  

  g_free (root_path);
  
  return g_file_test (*model_file, G_FILE_TEST_EXISTS);
}

/**
 * @brief internal function to get the cat.png
 */
static gboolean
_GetCatPngFilePath (gchar **input_file)
{
  const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();
  std::string input_file_name = "orange.png";

  *input_file = g_build_filename (
      root_path, "tests", "test_models", "data", input_file_name.c_str (), NULL);

  g_free (root_path);

  return g_file_test (*input_file, G_FILE_TEST_EXISTS);
}

/**
 * @brief Set tensor filter properties
 */
static void
_SetFilterProp (GstTensorFilterProperties *prop, const gchar *name, const gchar **models)
{
  memset (prop, 0, sizeof (GstTensorFilterProperties));
  prop->fwname = name;
  prop->fw_opened = 0;
  prop->model_files = models;
  prop->num_models = g_strv_length ((gchar **) models);
}

/**
 * @brief Signal to validate the result in tensor_sink
 */
static void
check_output (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  GstMemory *mem_res;
  GstMapInfo info_res;
  gboolean mapped;
  UNUSED (element);

  mem_res = gst_buffer_get_memory (buffer, 0);
  mapped = gst_memory_map (mem_res, &info_res, GST_MAP_READ);
  ASSERT_TRUE (mapped);

  gint is_float = (gint) * ((guint8 *) user_data);
  guint idx, max_idx = -1;

  if (is_float == 0) {
    guint8 *output = (guint8 *) info_res.data;
    guint8 max_value = 0;

    for (idx = 0; idx < info_res.size; ++idx) {
      if (output[idx] > max_value) {
        max_value = output[idx];
        max_idx = idx;
      }
    }
  } else if (is_float == 1) {
    gfloat *output = (gfloat *) info_res.data;
    gfloat max_value = G_MINFLOAT;
          
    for (idx = 0; idx < (info_res.size / sizeof (gfloat)); ++idx) {

      if (output[idx] > max_value) {
        
        max_value = output[idx];
        max_idx = idx;
      }
    }
  } else {
    ASSERT_TRUE (1 == 0);
  }

  EXPECT_EQ (max_idx, 282U);
}


/**
 * @brief Negative test case with invalid model file path
 */
TEST (nnstreamerFilterOnnxRuntime, openClose00)
{
  int ret;
  void *data = NULL;

  const gchar *model_files[] = {
    "some/invalid/model/path.onnx",
    NULL,
  };
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("onnxruntime");
  EXPECT_NE (sp, nullptr);
  GstTensorFilterProperties prop;

  _SetFilterProp (&prop, "onnxruntime", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);

}

/**
 * @brief Positive case with successful getModelInfo
 */
TEST (nnstreamerFilterOnnxRuntime, getModelInfo00)
{
  int ret;
  void *data = NULL;
  gchar *model_file;

  ASSERT_TRUE (_GetModelFilePath (&model_file));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("onnxruntime");
  EXPECT_NE (sp, nullptr);

  GstTensorFilterProperties prop;

  _SetFilterProp (&prop, "onnxruntime", model_files);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
}

/**
 * @brief Positive case with successful getModelInfo
 */
TEST (nnstreamerFilterOnnxRuntime, getModelInfo00_1)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  
  ASSERT_TRUE (_GetModelFilePath (&model_file));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("onnxruntime");
  EXPECT_NE (sp, nullptr);

  GstTensorFilterProperties prop;

  _SetFilterProp (&prop, "onnxruntime", model_files);

  ret = sp->open (&prop, &data);

  EXPECT_EQ (ret, 0);

  GstTensorsInfo in_info, out_info;
  ret = sp->getModelInfo (NULL, NULL, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);

  EXPECT_EQ (in_info.num_tensors, 1U);
  EXPECT_EQ (in_info.info[0].dimension[0], 224U);
  EXPECT_EQ (in_info.info[0].dimension[1], 224U);
  EXPECT_EQ (in_info.info[0].dimension[2], 3U);
  EXPECT_EQ (in_info.info[0].dimension[3], 1U);
  EXPECT_EQ (in_info.info[0].type, _NNS_FLOAT32);

  EXPECT_EQ (out_info.num_tensors, 1U);
  EXPECT_EQ (out_info.info[0].dimension[0], 1001U);
  EXPECT_EQ (out_info.info[0].dimension[1], 1U);
  EXPECT_EQ (out_info.info[0].dimension[2], 0U);
  EXPECT_EQ (out_info.info[0].dimension[3], 0U);
  EXPECT_EQ (out_info.info[0].type, _NNS_FLOAT32);

  sp->close (&prop, &data);

  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  g_free (model_file);
}

/**
 * @brief Test onnxruntime subplugin with successful invoke for sample onnx model (input data type: float)
 */
TEST (nnstreamerFilterOnnxRuntime, invoke00)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  gchar *model_file;
  GstTensorFilterProperties prop;

  ASSERT_TRUE (_GetModelFilePath (&model_file));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("onnxruntime");
  ASSERT_TRUE (sp != nullptr);
  _SetFilterProp (&prop, "onnxruntime", model_files);

  input.size = sizeof (float) * 224 * 224 * 3 * 1;
  output.size = sizeof (float) * 1001 * 1;

  input.data = g_malloc0 (input.size);
  output.data = g_malloc0 (output.size);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

    /** invoke successful */
  ret = sp->invoke (NULL, NULL, data, &input, &output);
  EXPECT_EQ (ret, 0);

  g_free (input.data);
  g_free (output.data);
  
  sp->close (&prop, &data);

  g_free (model_file);
}


/**
 * @brief Negative case with invalid input/output
 */
TEST (nnstreamerFilterOnnxRuntime, invoke01_n)
{
  int ret;
  void *data = NULL;
  GstTensorMemory input, output;
  gchar *model_file;
  GstTensorFilterProperties prop;

  ASSERT_TRUE (_GetModelFilePath (&model_file));

  const gchar *model_files[] = {
    model_file,
    NULL,
  };

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("onnxruntime");
  ASSERT_TRUE (sp != nullptr);
  _SetFilterProp (&prop, "onnxruntime", model_files);

  output.size = input.size = sizeof (float) * 1;
  input.data = g_malloc0 (input.size);
  output.data = g_malloc0 (output.size);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);

  /* catching exception */
  EXPECT_NE (sp->invoke (NULL, NULL, data, NULL, &output), 0);
  EXPECT_NE (sp->invoke (NULL, NULL, data, &input, NULL), 0);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
  g_free (model_file);
}


/**
 * @brief Negative case to launch gst pipeline: wrong dimension
 */
TEST (nnstreamerFilterOnnxRuntime, launch00_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  gchar *model_file;
  ASSERT_TRUE (_GetModelFilePath (&model_file));

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc num-buffers=10 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=42,height=42,framerate=0/1 ! tensor_converter ! tensor_filter framework=onnxruntime model=\"%s\" latency=1 ! tensor_sink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_free (model_file);
}

/**
 * @brief Negative case to launch gst pipeline: wrong data type
 */
TEST (nnstreamerFilterOnnxRuntime, launch01_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  gchar *model_file;
  ASSERT_TRUE (_GetModelFilePath (&model_file));

  /* create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc num-buffers=10 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_filter framework=onnxruntime model=\"%s\" latency=1 ! tensor_sink",
      model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  EXPECT_NE (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_free (model_file);
}

/**
 * @brief Positive case to launch gst pipeline
 */
TEST (nnstreamerFilterOnnxRuntime, floatModelResult)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  gchar *model_file, *input_file;

  ASSERT_TRUE (_GetModelFilePath (&model_file));
  ASSERT_TRUE (_GetCatPngFilePath (&input_file));

  //create a nnstreamer pipeline
  pipeline = g_strdup_printf ("filesrc location=\"%s\" ! pngdec ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_transform mode=arithmetic option=typecast:float32,div:127.5,add:-1.0 ! tensor_filter framework=onnxruntime model=\"%s\" ! tensor_sink name=sink",
      input_file, model_file);

  gstpipe = gst_parse_launch (pipeline, &err);
  ASSERT_TRUE (gstpipe != nullptr);

  GstElement *sink_handle = gst_bin_get_by_name (GST_BIN (gstpipe), "sink");

  ASSERT_TRUE (sink_handle != nullptr);

  guint8 *is_float = (guint8 *) g_malloc0 (1);
  *is_float = 1;

  g_signal_connect (sink_handle, "new-data", (GCallback) check_output, is_float);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_PLAYING, UNITTEST_STATECHANGE_TIMEOUT * 10),
      0);

  EXPECT_EQ (setPipelineStateSync (gstpipe, GST_STATE_NULL, UNITTEST_STATECHANGE_TIMEOUT), 0);

  gst_object_unref (sink_handle);
  gst_object_unref (gstpipe);
  g_free (pipeline);
  g_free (model_file);
  g_free (input_file);
  g_free (is_float);
}

/**
 * @brief Negative case with incorrect path
*/
TEST (nnstreamerFilterOnnxRuntime, error00_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  int status = 0;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *model_file = g_build_filename (root_path, "tests", "test_models",
      "models", "incorrect_path.onnx", NULL);
 
  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter framework=onnxruntime model=\"%s\" ! fakesink",
      model_file);
  gstpipe = gst_parse_launch (pipeline, &err);

  if (gstpipe) {
    status = 0;

    EXPECT_NE (gst_element_set_state (gstpipe, GST_STATE_PLAYING), GST_STATE_CHANGE_SUCCESS);
    EXPECT_EQ (gst_element_set_state (gstpipe, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);
    g_usleep (500000);
    EXPECT_NE (gst_element_set_state (gstpipe, GST_STATE_NULL), GST_STATE_CHANGE_FAILURE);
    g_usleep (100000);

    gst_object_unref (gstpipe);
  } else {
    status = -1;
    g_printerr ("GST PARSE LAUNCH FAILED: [%s], %s\n", pipeline,
        (err) ? err->message : "unknown reason");
    g_clear_error (&err);
  }
  EXPECT_EQ (status, 0);

  g_free (model_file);
  g_free (pipeline);
}

/**
 * @brief Negative case with incorrect tensor meta
*/
TEST (nnstreamerFilterOnnxRuntime, error01_n)
{
  gchar *pipeline;
  GstElement *gstpipe;
  GError *err = NULL;
  int status = 0;
  gchar *test_model;

  ASSERT_TRUE (_GetModelFilePath (&test_model));

  /* Create a nnstreamer pipeline */
  pipeline = g_strdup_printf ("videotestsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,format=RGB,width=240,height=224 ! tensor_converter ! tensor_filter framework=onnxruntime model=\"%s\" ! fakesink",
      test_model);

  gstpipe = gst_parse_launch (pipeline, &err);
  if (gstpipe) {
    status = 0;
    GstState state, pending;

    EXPECT_NE (gst_element_set_state (gstpipe, GST_STATE_PLAYING), GST_STATE_CHANGE_SUCCESS);
    g_usleep (500000);
    EXPECT_EQ (gst_element_get_state (gstpipe, &state, &pending, GST_SECOND / 4),
        GST_STATE_CHANGE_FAILURE);  //This should fail: dimension mismatched.

    EXPECT_NE (gst_element_set_state (gstpipe, GST_STATE_NULL), GST_STATE_CHANGE_FAILURE);
    g_usleep (100000);

    gst_object_unref (gstpipe);
  } else {
    status = -1;
    g_printerr ("GST PARSE LAUNCH FAILED: [%s], %s\n", pipeline,
        (err) ? err->message : "unknown reason");
    g_clear_error (&err);
  }
  EXPECT_EQ (status, 0);
  g_free (test_model);
  g_free (pipeline);
}

/**
 * @brief Main GTest
 */
int
main (int argc, char **argv)
{
  int result = -1;

  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  gst_init (&argc, &argv);

  /* Force the binary to use dlog_print of untitest-util by calling it directly */
  ml_logd ("onnxruntime test starts w/ dummy backend.");

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
