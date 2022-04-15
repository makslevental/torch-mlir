//
// Created by mlevental on 4/18/22.
//

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace std;

const int NUM_THREADS = 2;

vector<string> read_ll_file(string fp) {
  ifstream myfile(fp);
  string line;
  vector<string> lines;
  if (myfile.is_open()) {
    while (getline(myfile, line)) {
      lines.emplace_back(line);
    }
    myfile.close();
  }

  return lines;
}

const regex reg1(", align \\d+");
const regex reg2("(%[\\d|a-z|_]*|([0-9]*[.])+[0-9]+)");
const regex reg3("(float (%[\\d|a-z|_]*))");
const regex reg4("float ");
const regex reg5("(float\\* (%[\\d|a-z|_]*))");
const regex reg6("float\\* ");

std::pair<vector<string>, vector<string>> get_ssas_from_ir_line(string line) {

  line = std::regex_replace(line, reg1, "");
  std::string line_orig = line;

  vector<string> idents;
  std::smatch match;

  while (regex_search(line, match, reg2)) {
    idents.emplace_back(match[0]);
    line = match.suffix();
  }

  if (idents.empty())
    return std::make_pair(vector<string>{}, vector<string>{});

  vector<string> assign;
  vector<string> deps;
  if (line_orig.find("fmul") != std::string::npos ||
      line_orig.find("fadd") != std::string::npos ||
      line_orig.find("fcmp") != std::string::npos ||
      line_orig.find("fsub") != std::string::npos ||
      line_orig.find("fdiv") != std::string::npos) {
    assign = {idents[0]};
    deps = std::vector<string>(idents.begin() + 1, idents.end());
  } else if (line_orig.find("store") != std::string::npos) {
    assign = {idents[1]};
    deps = vector<string>{idents[0]};
  } else if (line_orig.find("expf") != std::string::npos) {
    assign = {idents[0]};
    deps = vector<string>{idents[1]};
  } else if (line_orig.find("define void @forward") != std::string::npos) {
    std::smatch match;
    while (regex_search(line_orig, match, reg3)) {
      auto inp = std::regex_replace(string(match[0]), reg4, ", ");
      assign.emplace_back(inp);
      line_orig = match.suffix();
    }
    while (regex_search(line_orig, match, reg5)) {
      auto outp = std::regex_replace(string(match[0]), reg6, ", ");
      deps.emplace_back(outp);
      line_orig = match.suffix();
    }
  } else {
    throw std::invalid_argument("wtfbbq");
  }

  return std::make_pair(assign, deps);
}

vector<vector<string>> topo_sort(map<string, set<string>> G) {
  vector<set<string>> todo_sets(NUM_THREADS);
  map<string, int> node_to_thread;

  int i = 0;
  for (auto [u, vs] : G) {
    todo_sets[i % NUM_THREADS].insert(u);
    node_to_thread[u] = i++ % NUM_THREADS;
    for (const auto &v : vs) {
      todo_sets[i % NUM_THREADS].insert(v);
      node_to_thread[v] = i++ % NUM_THREADS;
    }
  }

  vector<vector<string>> todos(NUM_THREADS);
  for (int j = 0; j < NUM_THREADS; ++j) {
    todos.emplace_back(todo_sets[j].begin(), todo_sets[j].end());
  }



  std::mutex mtx;
  std::vector<std::thread> threads;
  std::condition_variable cv;
  std::atomic<int> finished{0};
  std::atomic<int> initialized{0};
  std::atomic<bool> started{false};
  vector<vector<string>> all_outs;


  auto run = [&](int thread_id) {
    ++initialized;
    cv.notify_all();
    {
      std::unique_lock<std::mutex> m(mtx);
      cv.wait(m, [&]() -> bool { return started; });
    }

    auto todo_set = todo_sets[thread_id];
    auto todo = todos[thread_id];
    while (!todo_set.empty()) {
//      set<string> output;
      for (const auto &node : todo) {
        set<string> intersect;
        set_intersection(todo_set.begin(), todo_set.end(), G[node].begin(),
                         G[node].end(),
                         std::inserter(intersect, intersect.begin()));
        if (intersect.empty())
          output.insert(node);
      }

      set<string> set_diff;
      std::set_difference(todo_set.begin(), todo_set.end(), output.begin(),
                          output.end(),
                          std::inserter(set_diff, set_diff.begin()));
      todo_set = set_diff;

      vector<string> new_todo;
      std::copy_if(todo.begin(), todo.end(), std::back_inserter(new_todo),
                   [&todo_set](auto u) { return todo_set.count(u); });
      todo = new_todo;

      all_outs.emplace_back(vector<string>(output.begin(), output.end()));
    }

    ++finished;
    cv.notify_all();

  };



  threads.reserve(NUM_THREADS);
  for (int thread_id = 0; thread_id < NUM_THREADS; ++thread_id) {
    threads.emplace_back(run, thread_id);
  }
  {
    std::unique_lock<std::mutex> m(mtx);
    cv.wait(m, [&]() { return initialized == NUM_THREADS; });
  }

  started = true;
  cv.notify_all();

  if (!threads.empty()) {
    for (int thread_id = 0; thread_id < NUM_THREADS; ++thread_id) {
      threads[thread_id].join();
    }
  }

















//  vector<string> todo(todo_set.begin(), todo_set.end());

//  while (!todo_set.empty()) {
//    set<string> output;
//    for (const auto &node : todo) {
//      set<string> intersect;
//      set_intersection(todo_set.begin(), todo_set.end(), edges[node].begin(),
//                       edges[node].end(),
//                       std::inserter(intersect, intersect.begin()));
//      if (intersect.empty())
//        output.insert(node);
//    }
//
//    set<string> set_diff;
//    std::set_difference(todo_set.begin(), todo_set.end(), output.begin(),
//                        output.end(),
//                        std::inserter(set_diff, set_diff.begin()));
//    todo_set = set_diff;
//
//    vector<string> new_todo;
//    std::copy_if(todo.begin(), todo.end(), std::back_inserter(new_todo),
//                 [&todo_set](auto u) { return todo_set.count(u); });
//    todo = new_todo;
//
//    all_outs.emplace_back(vector<string>(output.begin(), output.end()));
//  }

  return all_outs;
}

tuple<map<string, set<string>>, map<string, string>, vector<string>,
      vector<string>>
make_graph(const std::vector<string> &lines) {
  map<string, set<string>> G;
  map<string, string> defining_lines;
  vector<string> inputs;
  vector<string> outputs;

  for (const auto &line : lines) {
    auto [assign, deps] = get_ssas_from_ir_line(line);
    if (assign.empty())
      continue;

    if (line.find("define void @forward") != std::string::npos) {
      inputs = assign;
      outputs = deps;
      continue;
    }

    if (line.find("declare") != std::string::npos) {
      continue;
    }

    defining_lines[assign[0]] = line;
    for (const auto &dep : deps)
      G[assign[0]].emplace(dep);
  }

  return std::make_tuple(G, defining_lines, inputs, outputs);
}

void make_stages(map<string, set<string>> G, map<string, string> defining_lines,
                 vector<string> inputs, vector<string> outputs) {


}

int main(int argc, char *argv[]) {

  auto fp = string(argv[1]);
  auto lines = read_ll_file(fp);
  auto [G, defining_lines, inputs, outputs] = make_graph(lines);

  auto topo_sort_G = topo_sort(G);


  //  vector<tuple<string, string>> tuples;
  //
  //  tuples.emplace_back("11", "5");
  //  tuples.emplace_back("11", "7");
  //  tuples.emplace_back("8", "3");
  //  tuples.emplace_back("2", "11");
  //  tuples.emplace_back("9", "11");
  //  tuples.emplace_back("9", "8");
  //  tuples.emplace_back("10", "11");
  //
  //  auto top_sort = topo_sort(tuples);
  //  for (int i = 0; i < top_sort.size(); ++i) {
  //    cout << "stage " << i << ": ";
  //    for (const auto &item : top_sort[i])
  //      cout << item << ", ";
  //    cout << "\n";
  //  }

  return 0;
}