#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/jit/ir/irparser.h> // for parseIR
#include <torch/csrc/jit/jit_log.h> // getHeader
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/remove_mutation.h> // RemoveMutation
#include <torch/csrc/jit/passes/tensorexpr_fuser.h> // RemoveProfileNodesAndSpecializeTypes
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>
#include <torch/csrc/jit/runtime/profiling_record.h>
#include <torch/script.h>

using namespace torch::jit;

std::shared_ptr<Graph> getGraphFromModule(torch::jit::script::Module module) {
  std::shared_ptr<Module> module_ptr;
  module.eval();
  module_ptr = std::make_shared<Module>(freeze_module(module));
  auto forward = module_ptr->get_method("forward");
  auto graph = forward.graph();
  graph->eraseInput(0);

  torch::jit::Inline(*graph);
  torch::jit::RemoveTensorMutation(graph);
  return graph;
}

std::shared_ptr<Graph> traceWithShapes(
    std::shared_ptr<Graph> g,
    const py::tuple& input_tuple,
    std::string model_name) {
  // run once to type info
  auto stack = toTraceableStack(input_tuple);

//  torch::jit::Inline(*g);
  torch::jit::RemoveTensorMutation(g);
  auto pr = torch::jit::ProfilingRecord::instrumentGraph(g);
  auto graph = pr->profiled_graph_;

  Code cd(graph, model_name);
  InterpreterState is{cd};
  is.run(stack);

  // plan
  ProfilingRecord::removeProfileCounter(graph->block());
  torch::jit::RemoveProfileNodesAndSpecializeTypes(graph);
  return graph;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("traceWithShapes", &traceWithShapes, "");
}
