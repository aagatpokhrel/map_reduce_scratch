#include "mapreduce.h"
#include <omp.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <dirent.h>
#include <cstring>
#include <algorithm>
#include <atomic>

#define NUM_REDUCERS 4
#define NUM_MAPPERS 4
#define NUM_READERS 8
#define PERCENTAGE 1.0
std::vector<double> reader_times(NUM_READERS, 0.0);
std::vector<double> mapper_times(NUM_MAPPERS, 0.0);
std::vector<double> reducer_times(NUM_REDUCERS, 0.0);
double reader_time = 0.0;
unsigned long long total_words = 0;

bool has_txt_extension(const std::string &s) {
    if (s.length() >= 4) {
        return (0 == s.compare(s.length() - 4, 4, ".txt"));
    }
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
    } else {
        std::cerr << "cant open directory: " << path << std::endl;
    }
}

void run_reader(int tid, const std::vector<std::string>& files, std::atomic<size_t>& file_idx_counter, SafeQueue<std::string>& line_queue) {
    // reader thread read data and place into work queue
    double start = omp_get_wtime();

    size_t idx;
    while ((idx = file_idx_counter.fetch_add(1)) < files.size()) {
        std::ifstream file(files[idx]);
        std::string line;
        if (file.is_open()) {
            while (std::getline(file, line)) {
                line_queue.push(line);
            }
            file.close();
        }
    }
    double end = omp_get_wtime();
    reader_times[tid] = end - start;
}

void run_mapper(int tid, SafeQueue<std::string>& line_queue, 
                std::vector<SafeQueue<std::pair<std::string, int>>>& reducer_queues) {
    double start = omp_get_wtime();
    std::string line;
    // local buffer for optimization
    std::map<std::string, int> local_counts; 
    unsigned long long local_word_count = 0;
    
    while (line_queue.pop(line)) {
        std::stringstream ss(line);
        std::string word;
        while (ss >> word) {
            local_counts[word]++;
            local_word_count++;
        }
    }

    #pragma omp atomic
    total_words += local_word_count;

    // combined result to reducer
    for (auto const& [key, val] : local_counts) {
        int r_id = partition_key(key, NUM_REDUCERS);
        reducer_queues[r_id].push({key, val});
    }
    double end = omp_get_wtime();
    mapper_times[tid] = end - start;
}

void run_reducer(int tid, SafeQueue<std::pair<std::string, int>>& my_queue) {
    double start = omp_get_wtime();
    std::map<std::string, int> final_counts;
    std::pair<std::string, int> item;

    // collect produced by mappers
    while (my_queue.pop(item)) {
        final_counts[item.first] += item.second;
    }

    double end = omp_get_wtime();
    reducer_times[tid] = end - start;

    // output file, i dont count i/o here
    std::ofstream outfile("output_reducer_" + std::to_string(tid) + ".txt");
    for (auto const& [key, val] : final_counts) {
        outfile << key << " " << val << "\n";
    }
    outfile.close();
}

int main(int argc, char** argv) {
    omp_set_nested(1);
    std::string input_dir = "raw_text_input";
    std::vector<std::string> all_files;

    get_txt_files(input_dir, all_files);

    // to get subset of files so that we operate on different loads for analysis
    size_t num_files_to_use = static_cast<size_t>(all_files.size() * PERCENTAGE);
    if (num_files_to_use < 1) num_files_to_use = 1;
    
    std::vector<std::string> files(all_files.begin(), all_files.begin() + num_files_to_use);

    SafeQueue<std::string> line_queue;
    std::vector<SafeQueue<std::pair<std::string, int>>> reducer_queues(NUM_REDUCERS);
    std::atomic<size_t> file_idx_counter(0); //readers to claim for files'

    double start_time = omp_get_wtime();

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            #pragma omp parallel num_threads(NUM_READERS)
            {
                int tid = omp_get_thread_num();
                run_reader(tid, files, file_idx_counter, line_queue);
            }
            line_queue.setFinished(); // this notifies for mappers

        }
        #pragma omp section
        {
            #pragma omp parallel num_threads(NUM_MAPPERS) 
            {
                int tid = omp_get_thread_num();
                run_mapper(tid, line_queue, reducer_queues);
            }
            for(int i=0; i<NUM_REDUCERS; i++) {
                reducer_queues[i].setFinished();
            }
        }
        #pragma omp section
        {
            #pragma omp parallel num_threads(NUM_REDUCERS)
            {
                int tid = omp_get_thread_num();
                if (tid < NUM_REDUCERS) {
                    run_reducer(tid, reducer_queues[tid]);
                }
            }
        }
    }
    double end_time = omp_get_wtime();

    double max_read = 0, avg_read = 0, min_read = reader_times[0];
    for (double t : reader_times) {
        if (t > max_read) max_read = t;
        if (t < min_read) min_read = t;
        avg_read += t;
    }
    avg_read /= NUM_READERS;

    double max_map = 0, avg_map = 0, min_map = mapper_times[0];
    for(double t : mapper_times) {
        if(t > max_map) max_map = t; 
        if(t < min_map) min_map = t;
        avg_map += t; 
    }
    avg_map = avg_map/NUM_MAPPERS;
    
    double max_red = 0, avg_red = 0, min_red = reducer_times[0];
    for(double t : reducer_times) {
        if(t > max_red) max_red = t; 
        if(t < min_red) min_red = t;
        avg_red += t;
    }
    avg_red /= NUM_REDUCERS;

    std::cout << "Total Execution Time "<< end_time - start_time << "\n";
    std::cout << "Total Words Evaluated "<< total_words << "\n";
    std::cout << "Reader Time \tAverage:  " << avg_read << " s (Max: " << max_read << " s)  (Min:" << min_read << " s)\n";
    std::cout << "Mapper Time \tAverage:  " << avg_map << " s (Max: " << max_map << " s)  (Min:" << min_map << " s)\n";
    std::cout << "Reducer Time \tAverage: " << avg_red << " s (Max: " << max_red << " s)  (Min:" << min_red << " s)\n";
    for(int i=0; i<NUM_READERS; i++) std::cout << "Reader " << i << " Time: " << reader_times[i] << " s\n";
    for(int i=0; i<NUM_MAPPERS; i++) std::cout << "Mapper " << i << " Time: " << mapper_times[i] << " s\n";
    for(int i=0; i<NUM_REDUCERS; i++) std::cout << "Reducer " << i << "Time : " << reducer_times[i] << " s\n";
    return 0;
}