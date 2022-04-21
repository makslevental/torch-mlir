//
// Created by mlevental on 4/18/22.
//

#include "thread_pool.hpp"
#include <functional>
#include <numeric>
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
#include "yamc_barrier.hpp"
#include <thread>
#include <vector>

using namespace std;

const unsigned long NUM_THREADS = 8;

template<typename... T>
void print(const T &...items) {
  (cerr << ... << items);
}

template<typename... T>
void println(const T &...items) {
  print(items..., '\n');
}

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

std::pair<vector<string>, vector<string>> get_ssas_from_ir_line(string line, synced_stream &sync_out) {
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
  if (line_orig.find("fmul") != std::string::npos || line_orig.find("fadd") != std::string::npos ||
      line_orig.find("fcmp") != std::string::npos || line_orig.find("fsub") != std::string::npos ||
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
      auto inp = std::regex_replace(string(match[0]), reg4, "");
      assign.emplace_back(inp);
      line_orig = match.suffix();
    }
    while (regex_search(line_orig, match, reg5)) {
      auto outp = std::regex_replace(string(match[0]), reg6, "");
      deps.emplace_back(outp);
      line_orig = match.suffix();
    }
  } else {
    throw std::invalid_argument("wtfbbq");
  }

  return std::make_pair(assign, deps);
}

template<typename Key, typename Val>
struct ThreadSafeMap {
  std::map<Key, Val> mainCache;
  std::map<Key, std::unique_ptr<std::mutex>> mutexCache;
  std::mutex gMutex;

  Val getSafe(Key key) {
    std::mutex *inner_mutex;
    {
      std::lock_guard<std::mutex> g_lk(gMutex);
      auto it = mutexCache.find(key);
      if (it == mutexCache.end()) {
        it = mutexCache.emplace(key, std::make_unique<std::mutex>()).first;
      }
      inner_mutex = it->second.get();
    }
    {
      std::lock_guard<std::mutex> c_lk(*inner_mutex);
      return mainCache[key];
    }
  }

  void decrSafe(Key key) {
    std::mutex *inner_mutex;
    {
      std::lock_guard<std::mutex> g_lk(gMutex);
      auto it = mutexCache.find(key);
      if (it == mutexCache.end()) {
        it = mutexCache.emplace(key, std::make_unique<std::mutex>()).first;
      }
      inner_mutex = it->second.get();
    }
    {
      std::lock_guard<std::mutex> c_lk(*inner_mutex);
      mainCache[key]--;
    }
  }

  Val getUnsafe(Key key) { return mainCache[key]; }

  void setSafe(Key key, Val val) {
    std::mutex *inner_mutex;
    {
      std::lock_guard<std::mutex> g_lk(gMutex);
      auto it = mutexCache.find(key);
      if (it == mutexCache.end()) {
        it = mutexCache.emplace(key, std::make_unique<std::mutex>()).first;
      }
      inner_mutex = it->second.get();
    }
    {
      std::lock_guard<std::mutex> c_lk(*inner_mutex);
      mainCache[key] = val;
    }
  }

  void setUnsafe(Key key, Val val) { mainCache[key] = val; }

  void print() {
    for (const auto &item: mainCache)
      println(item.first, " in degree ", item.second);
  }
};

typedef std::uint_fast32_t ui32;
typedef std::uint_fast64_t ui64;

template<typename T1, typename T2, typename F>
void
parallelize_loop(thread_pool &pool, const T1 &first_index, const T2 &index_after_last, F &loop, ui32 num_blocks = 0,
                 ui32 sleep_duration = 1) {
  typedef std::common_type_t<T1, T2> T;
  T the_first_index = (T) first_index;
  T last_index = (T) index_after_last;
  if (the_first_index == last_index)
    return;
  if (last_index < the_first_index) {
    T temp = last_index;
    last_index = the_first_index;
    the_first_index = temp;
  }
  last_index--;
  if (num_blocks == 0)
    num_blocks = NUM_THREADS;
  ui64 total_size = (ui64) (last_index - the_first_index + 1);
  ui64 block_size = (ui64) (total_size / num_blocks);
  if (block_size == 0) {
    block_size = 1;
    num_blocks = (ui32) total_size > 1 ? (ui32) total_size : 1;
  }
  std::atomic<ui32> blocks_running = 0;
  for (ui32 t = 0; t < num_blocks; t++) {
    T start = ((T) (t * block_size) + the_first_index);
    T end = (t == num_blocks - 1) ? last_index + 1 : ((T) ((t + 1) * block_size) + the_first_index);
    blocks_running++;
    pool.push_task([start, end, &loop, &blocks_running] {
      loop(start, end);
      blocks_running--;
    });
  }
  while (blocks_running != 0) {
    if (sleep_duration)
      std::this_thread::sleep_for(std::chrono::microseconds(sleep_duration));
    else
      std::this_thread::yield();
  }
}


tuple<vector<string>, map<string, set<string>>, map<string, set<string>>, map<string, string>, vector<string>, vector<string>>
make_graph(const std::vector<string> &lines) {
  map<string, set<string>> G;
  map<string, set<string>> Ginv;
  map<string, string> defining_lines;
  vector<string> inputs;
  vector<string> outputs;
  thread_pool pool(NUM_THREADS);
  synced_stream sync_out;

  auto loop = [&lines, &inputs, &outputs, &defining_lines, &G, &Ginv, &sync_out](const int &a, const int &b) mutable {
    for (int i = a; i < b; ++i) {
      auto line = lines[i];
      auto [assign, deps] = get_ssas_from_ir_line(line, sync_out);
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
      for (const auto &assign: assign) {
        for (const auto &dep: deps) {
          // dep precedes assign
          G[dep].emplace(assign);
          Ginv[assign].emplace(dep);
        }
      }
    }
  };
  parallelize_loop(pool, 0, lines.size(), loop);
  pool.wait_for_tasks();
  for (const auto &inp: inputs) {
    Ginv[inp] = {};
  }
  for (const auto &outp: outputs) {
    G[outp] = {};
  }

  set<string> all_set{};
  for (const auto &item: G) {
    all_set.insert(item.first);
    for (const auto &item2: item.second)
      all_set.insert(item2);
  }
  for (const auto &item: Ginv) {
    all_set.insert(item.first);
    for (const auto &item2: item.second)
      all_set.insert(item2);
  }

  //    for (const auto &item: G) {
  //        print("dep ", item.first, " assigns :");
  //        for (const auto &item2: item.second)
  //            print(item2, ", ");
  //        println("");
  //    }
  //    for (const auto &item: Ginv) {
  //        print("assign ", item.first, " deps :");
  //        for (const auto &item2: item.second)
  //            print(item2, ", ");
  //        println("");
  //    }

  auto all_vec = vector<string>(all_set.begin(), all_set.end());
  std::sort(all_vec.begin(), all_vec.end());
  return std::make_tuple(all_vec, G, Ginv, defining_lines, inputs, outputs);
}

vector<set<std::string>>
parallel_topo_sort(vector<string> all, map<string, set<string>> G, map<string, set<string>> Ginv) {
  ThreadSafeMap<string, int> in_degree;
  thread_pool pool(NUM_THREADS);

  // initialize all in-degrees
  // note this is the in-degree from u <- v
  for (auto &u: all) {
    if (Ginv.count(u))
      in_degree.setUnsafe(u, Ginv[u].size());
    else
      in_degree.setUnsafe(u, 0);
  }
  in_degree.print();


  // partition nodes
  size_t length = all.size() / NUM_THREADS;
  size_t remain = all.size() % NUM_THREADS;
  size_t begin = 0;
  size_t end = 0;
  map<int, set<string>> partition;
  for (size_t i = 0; i < std::min(NUM_THREADS, all.size()); ++i) {
    end += (remain > 0) ? (length + !!(remain--)) : length;
    partition[i] = set<string>(all.begin() + begin, all.begin() + end);
    println("partition ", i, " ", partition[i].size());
    begin = end;
  }
  println("\n");

  // initialize 0 in degree nodes per thread/partition
  atomic<int> depth = 0;
  atomic<int> num_not_done = NUM_THREADS;

  map<int, map<int, set<string>>> all_outputs;
  yamc::barrier sync_point(NUM_THREADS, [&depth, &partition, &num_not_done, &in_degree]()mutable {
    println("\n");
    ++depth;
    int _num_not_done = 0;
    for (const auto &item: partition) {
      if (!item.second.empty())
        _num_not_done += 1;
      else {
        //                print(item.second.size(), ", ");
      }
    }
    //        for (const auto &item: partition) {
    //            if (item.second.size() < 20)
    //                for (const auto &item: item.second) {
    //                    print(item, " in degree ", in_degree.mainCache[item], ", ");
    //                }
    //            println("");
    //        }
    //        in_degree.print();

    num_not_done = _num_not_done;
    println("num done ", num_not_done);
    println("\n");
  });

  for (int i = 0; i < NUM_THREADS; ++i) {
    all_outputs[i] = {};
    pool.push_task([&in_degree, i, &all_outputs, &G, &partition, &depth, &sync_point, &num_not_done]() mutable {
      while (num_not_done > 0) {
        // find new 0 degree nodes
        sync_point.arrive_and_wait();
        set<string> output;
        for (const auto &u: partition[i]) {
          if (in_degree.getSafe(u) <= 0) {
            println(i, " outputting ", u);
            output.insert(u);
          }
        }

        sync_point.arrive_and_wait();
        for (const auto &outp: output) {
          for (const auto &assign: G[outp]) {
            auto deg = in_degree.getSafe(assign);
            in_degree.decrSafe(assign);
            auto degn = in_degree.getSafe(assign);
            println(i, " output ", outp, " decrementing ", assign, " from ", deg, " to ", degn);
          }
        }
        sync_point.arrive_and_wait();

        for (const auto &item: output) {
          println(i, " erasing from partition ", item);
          partition[i].erase(item);
        }

        all_outputs[i][depth] = output;

        this_thread::sleep_for(std::chrono::microseconds(1));
      }
    });
  }
  pool.wait_for_tasks();

  int _depth = depth;
  println("max depth ", _depth);
  vector<set<string>> stages;
  for (int d = 0; d < _depth; ++d) {
    set<string> set{};
    for (int i = 0; i < NUM_THREADS; ++i) {
      std::set_union(set.begin(), set.end(), all_outputs[i][d].begin(), all_outputs[i][d].end(),
                     inserter(set, set.end()));
    }
    if (!set.empty()) {
      stages.push_back(set);
    }
  }
  return stages;
}


vector<set<std::string>> topo_sort(set<string> all, map<string, set<string>> G, map<string, set<string>> Ginv) {
  map<string, int> in_degree;
  for (auto &u: all) {
    if (Ginv.count(u))
      in_degree.emplace(u, Ginv[u].size());
    else
      in_degree.emplace(u, 0);
  }

  vector<set<string>> all_outputs;

  int depth = 0;
  while (!all.empty()) {
    println(depth++);
    // find new 0 degree nodes
    set<string> output;
    for (const auto &u: all) {
      if (in_degree[u] <= 0)
        output.insert(u);
    }

    // update in degree
    for (const auto &outp: output) {
      for (const auto &assign: G[outp]) {
        in_degree[assign]--;
      }
      all.erase(outp);
    }
    if (!output.empty())
      all_outputs.push_back(output);
  }
  return all_outputs;
}

void make_stages(map<string, set<string>> G, map<string, string> defining_lines, vector<string> inputs,
                 vector<string> outputs) {}

int main(int argc, char *argv[]) {

  auto fp = string(argv[1]);
  auto lines = read_ll_file(fp);
  auto [all, G, Ginv, defining_lines, inputs, outputs] = make_graph(lines);

  //    auto stages = topo_sort(set<string>(all.begin(), all.end()), G, Ginv);
  auto stages = parallel_topo_sort(all, G, Ginv);
  cout << "num stages: " << stages.size() << "\n";
  for (int i = 0; i < stages.size(); ++i) {
    cout << "stage " << i << " ops " << stages[i].size() << "\n";
    for (const auto &item: stages[i])
      cout << item << " -- ";
    cout << "\n";
  }

  return 0;
}