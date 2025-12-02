#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <map>
#include <fstream>
#include <sstream>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <atomic>
#include <dirent.h>

#define NUM_READERS 4
#define NUM_MAPPERS 8
#define NUM_REDUCERS 4
#define BATCH_SIZE 500

#define TAG_FILE_REQ 1
#define TAG_FILE_RESP 2
#define TAG_DATA 3
#define TAG_TERMINATE 4
#define TAG_DONE_READING 5


struct WordPacket {
    char word[32];
    int count;
};

struct NodeMetrics {
    double total_runtime = 0.0;
    std::vector<double> reader_times;
    std::vector<double> mapper_times;
    std::vector<double> reducer_times;
    
    double sender_time = 0.0;
    double receiver_time = 0.0;
    unsigned long long total_words = 0;
    
    double io_time = 0.0;
    double map_proc_time = 0.0;
    double map_wait_time = 0.0;
    double map_finish_time = 0.0;
};
NodeMetrics g_metrics;
std::mutex g_metrics_mtx;

// this queue is slightly different than the mapreduce.h queue
template <typename T>
class SafeQueue {
private:
    std::queue<T> q;
    std::mutex mtx;
    std::condition_variable cv;
    bool finished = false;
public:
    void push(T val) {
        std::lock_guard<std::mutex> lock(mtx);
        q.push(val);
        cv.notify_one();
    }
    
    bool pop(T& val, double& wait_accumulator) {
        std::unique_lock<std::mutex> lock(mtx);
        if (q.empty() && !finished) {
            double start_wait = omp_get_wtime();
            while (q.empty() && !finished) {
                cv.wait(lock);
            }
            double end_wait = omp_get_wtime();
            wait_accumulator += (end_wait - start_wait);
        }
        if (q.empty() && finished) return false;
        val = q.front();
        q.pop();
        return true;
    }

    bool pop(T& val) {
        double dummy = 0;
        return pop(val, dummy);
    }

    void setFinished() {
        std::lock_guard<std::mutex> lock(mtx);
        finished = true;
        cv.notify_all();
    }
    
    bool isFinished() {
        std::lock_guard<std::mutex> lock(mtx);
        return finished && q.empty();
    }
};


int partition_key(const std::string& key, int num_nodes) {
    if (num_nodes <= 1) return 0;
    std::hash<std::string> hasher;
    int num_workers = num_nodes - 1;
    return (hasher(key) % num_workers) + 1;
}

int local_partition_key(const std::string& key, int num_reducers) {
    std::hash<std::string> hasher;
    return hasher(key) % num_reducers;
}

bool has_txt_extension(const std::string &s) {
    if (s.length() >= 4) return (0 == s.compare(s.length() - 4, 4, ".txt"));
    return false;
}

void get_txt_files(const std::string& path, std::vector<std::string>& files) {
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(path.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if (filename != "." && filename != ".." && has_txt_extension(filename)) {
                files.push_back(path + "/" + filename);
            }
        }
        closedir(dir);
    }
}

void master_node(int size, double percentage) {
    std::string input_dir = "raw_text_input";
    std::vector<std::string> all_files;
    get_txt_files(input_dir, all_files);

    size_t num_files = static_cast<size_t>(all_files.size() * percentage);
    if (num_files < 1) num_files = 1;
    if (num_files > all_files.size()) num_files = all_files.size();
    
    std::vector<std::string> files(all_files.begin(), all_files.begin() + num_files);

    size_t file_idx = 0;
    int finished_nodes = 0;
    int expected_nodes = size - 1;

    while (finished_nodes < expected_nodes) {
        MPI_Status status;
        int dummy;
        MPI_Recv(&dummy, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int source = status.MPI_SOURCE;
        int tag = status.MPI_TAG;

        if (tag == TAG_FILE_REQ) {
            if (file_idx < files.size()) {
                std::string f = files[file_idx++];
                MPI_Send(f.c_str(), f.size() + 1, MPI_CHAR, source, TAG_FILE_RESP, MPI_COMM_WORLD);
            } else {
                char end_sig = '\0';
                MPI_Send(&end_sig, 1, MPI_CHAR, source, TAG_FILE_RESP, MPI_COMM_WORLD);
            }
        } 
        else if (tag == TAG_DONE_READING) {
            finished_nodes++;
        }
    }
}

void worker_node(int rank, int size) {
    SafeQueue<std::string> line_queue;
    std::vector<SafeQueue<WordPacket>> outgoing_queues(size); 
    std::vector<SafeQueue<std::pair<std::string, int>>> reducer_queues(NUM_REDUCERS); 

    std::atomic<bool> local_mappers_done(false);
    std::atomic<int> active_readers(NUM_READERS); 

    {
        std::lock_guard<std::mutex> lock(g_metrics_mtx);
        g_metrics.reader_times.resize(NUM_READERS, 0.0);
        g_metrics.mapper_times.resize(NUM_MAPPERS, 0.0);
        g_metrics.reducer_times.resize(NUM_REDUCERS, 0.0);
    }

    omp_set_nested(1);
    double node_start = MPI_Wtime();

    #pragma omp parallel sections
    {
        // readers
        #pragma omp section
        {
            #pragma omp parallel num_threads(NUM_READERS)
            {
                int tid = omp_get_thread_num();
                double t_start = omp_get_wtime();
                double t_io_accum = 0;
                while (true) {
                    double t_io_start = omp_get_wtime();
                    int dummy = 0;
                    #pragma omp critical(mpi_req)
                    { MPI_Send(&dummy, 1, MPI_INT, 0, TAG_FILE_REQ, MPI_COMM_WORLD); }

                    char filename[256];
                    MPI_Status status;
                    MPI_Recv(filename, 256, MPI_CHAR, 0, TAG_FILE_RESP, MPI_COMM_WORLD, &status);
                    t_io_accum += (omp_get_wtime() - t_io_start);

                    if (filename[0] == '\0') break;

                    std::ifstream file(filename);
                    std::string line;
                    if(file.is_open()) {
                        while(std::getline(file, line)) line_queue.push(line);
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(g_metrics_mtx);
                    g_metrics.reader_times[tid] = omp_get_wtime() - t_start;
                    g_metrics.io_time += t_io_accum;
                }
                if (active_readers.fetch_sub(1) == 1) {
                    int dummy = 0;
                    #pragma omp critical(mpi_req) 
                    { MPI_Send(&dummy, 1, MPI_INT, 0, TAG_DONE_READING, MPI_COMM_WORLD); }
                }
            }
            line_queue.setFinished();
        }

        // mappers
        #pragma omp section
        {
            #pragma omp parallel num_threads(NUM_MAPPERS)
            {
                int tid = omp_get_thread_num();
                double t_start = omp_get_wtime();
                std::string line;
                unsigned long long local_words = 0;
                double my_wait = 0;

                while(line_queue.pop(line, my_wait)) {
                    std::stringstream ss(line);
                    std::string word;
                    while(ss >> word) {
                        local_words++;
                        if(word.size() > 31) word = word.substr(0, 31);
                        int target = partition_key(word, size);
                        if (target == rank) {
                            reducer_queues[local_partition_key(word, NUM_REDUCERS)].push({word, 1});
                        } else {
                            WordPacket p;
                            strncpy(p.word, word.c_str(), 31); p.word[31] = '\0'; p.count = 1;
                            outgoing_queues[target].push(p);
                        }
                    }
                }
                #pragma omp atomic
                g_metrics.total_words += local_words;
                
                double t_end = omp_get_wtime();
                {
                    std::lock_guard<std::mutex> lock(g_metrics_mtx);
                    g_metrics.mapper_times[tid] = t_end - t_start;
                    g_metrics.map_proc_time += (t_end - t_start - my_wait);
                    g_metrics.map_wait_time += my_wait;
                }
            }
            local_mappers_done = true;
            g_metrics.map_finish_time = MPI_Wtime();
            for(int i=0; i<size; i++) outgoing_queues[i].setFinished();
        }

        // sender
        #pragma omp section
        {
            double t_start = omp_get_wtime();
            bool working = true;
            std::vector<std::vector<WordPacket>> buffers(size);
            while(working) {
                working = false; 
                for(int i=0; i<size; i++) {
                    if (i == rank || i == 0) continue; 
                    WordPacket p;
                    while(outgoing_queues[i].pop(p)) {
                        working = true;
                        buffers[i].push_back(p);
                        if(buffers[i].size() >= BATCH_SIZE) {
                            #pragma omp critical(mpi_send)
                            { MPI_Send(buffers[i].data(), buffers[i].size() * sizeof(WordPacket), MPI_BYTE, i, TAG_DATA, MPI_COMM_WORLD); }
                            buffers[i].clear();
                        }
                    }
                    if(!outgoing_queues[i].isFinished()) working = true;
                }
            }
            for(int i=0; i<size; i++) {
                if(!buffers[i].empty()) {
                    #pragma omp critical(mpi_send)
                    { MPI_Send(buffers[i].data(), buffers[i].size() * sizeof(WordPacket), MPI_BYTE, i, TAG_DATA, MPI_COMM_WORLD); }
                }
            }
            
            // indicate termination by sending dummy
            int dummy_term = 0;
            for(int i=0; i<size; i++) {
                if(i != rank && i != 0) {
                     #pragma omp critical(mpi_send)
                     { MPI_Send(&dummy_term, 1, MPI_INT, i, TAG_TERMINATE, MPI_COMM_WORLD); }
                }
            }
            g_metrics.sender_time = omp_get_wtime() - t_start;
        }

        // receiver
        #pragma omp section
        {
            double t_start = omp_get_wtime();
            int nodes_finished = 0;
            int expected = size - 2; 
            if(expected < 0) expected = 0;

            while(nodes_finished < expected) {
                MPI_Status status;
                MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                if(status.MPI_TAG == TAG_TERMINATE) {
                    int dummy;
                    // should match the sender
                    MPI_Recv(&dummy, 1, MPI_INT, status.MPI_SOURCE, TAG_TERMINATE, MPI_COMM_WORLD, &status);
                    nodes_finished++;
                } else if(status.MPI_TAG == TAG_DATA) {
                    int count;
                    MPI_Get_count(&status, MPI_BYTE, &count);
                    std::vector<WordPacket> buf(count / sizeof(WordPacket));
                    MPI_Recv(buf.data(), count, MPI_BYTE, status.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD, &status);
                    for(auto& p : buf) {
                        reducer_queues[local_partition_key(p.word, NUM_REDUCERS)].push({std::string(p.word), p.count});
                    }
                }
            }
            while(!local_mappers_done) { 
                #pragma omp taskyield 
            }
            for(int i=0; i<NUM_REDUCERS; i++) reducer_queues[i].setFinished();
            g_metrics.receiver_time = omp_get_wtime() - t_start;
        }

        // reducers
        #pragma omp section
        {
            #pragma omp parallel num_threads(NUM_REDUCERS)
            {
                int tid = omp_get_thread_num();
                double t_start = omp_get_wtime();
                std::map<std::string, int> counts;
                std::pair<std::string, int> item;
                while(reducer_queues[tid].pop(item)) { counts[item.first] += item.second; }
                std::ofstream out("result_node_" + std::to_string(rank) + "_red_" + std::to_string(tid) + ".txt");
                for(auto const& pair : counts) out << pair.first << " " << pair.second << "\n";
                out.close();
                { std::lock_guard<std::mutex> lock(g_metrics_mtx); g_metrics.reducer_times[tid] = omp_get_wtime() - t_start; }
            }
        }
    }

    g_metrics.total_runtime = MPI_Wtime() - node_start;

    std::stringstream ss;
    ss << "\nNode: " << rank << " Details\n";
    ss << "  Total Runtime: " << g_metrics.total_runtime << "s\n";
    ss << "  Words Proc:    " << g_metrics.total_words << "\n";
    ss << "  Sender: " << g_metrics.sender_time << "s, Receiver: " << g_metrics.receiver_time << "s\n";
    ss << "  Readers: "; for(auto t : g_metrics.reader_times) ss << t << " ";
    ss << "\n  Mappers: "; for(auto t : g_metrics.mapper_times) ss << t << " ";
    ss << "\n  Reducers: "; for(auto t : g_metrics.reducer_times) ss << t << " ";
    ss << "\n";
    std::cout << ss.str() << std::endl; 
    
    double total_io, total_map, total_wait;
    double min_finish, max_finish;
    
    MPI_Reduce(&g_metrics.io_time, &total_io, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&g_metrics.map_proc_time, &total_map, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&g_metrics.map_wait_time, &total_wait, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&g_metrics.map_finish_time, &min_finish, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&g_metrics.map_finish_time, &max_finish, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 1) {
        int num_workers = size - 1;
        std::cout << "Avg Reader I/O Time: " << (total_io / num_workers) / NUM_READERS << " s\n";
        std::cout << "Avg Mapper Proc Time:" << (total_map / num_workers) / NUM_MAPPERS << " s\n";
        std::cout << "Avg Mapper Wait Time:" << (total_wait / num_workers) / NUM_MAPPERS << " s\n";
        std::cout << "Load Imbalance:      " << (max_finish - min_finish) << " s\n";
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) MPI_Abort(MPI_COMM_WORLD, 1);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double percentage = 1.0;
    if (argc > 1) percentage = std::stod(argv[1]);

    // 2 nodes at least
    if (rank == 0) master_node(size, percentage);
    else worker_node(rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}