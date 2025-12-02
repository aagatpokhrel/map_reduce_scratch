#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <chrono>

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
    } else {
        std::cerr << "cant open directory: " << path << std::endl;
    }
}

int main(){
    double percentage = 1.0;

    std::string input_dir = "raw_text_input";
    std::vector<std::string> all_files;
    get_txt_files(input_dir, all_files);

    size_t num_files_use = static_cast<size_t>(all_files.size() * percentage);
    if (num_files_use < 1) num_files_use = 1;
    std::vector<std::string> files(all_files.begin(), all_files.begin() + num_files_use);


    auto start_time = std::chrono::high_resolution_clock::now();

    std::map<std::string, int> word_counts;
    unsigned long long total_words = 0;

    for (const auto& filename : files) {
        std::ifstream file(filename);
        std::string line;
        if (file.is_open()) {
            while (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string word;
                while (ss >> word) {
                    word_counts[word]++;
                    total_words++;
                }
            }
            file.close();
        }
    }

    std::ofstream outfile("result_serial.txt");
    for (auto const& [key, val] : word_counts) {
        outfile << key << " " << val << "\n";
    }
    outfile.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "Sequential Time: " << std::fixed << std::setprecision(6) << duration.count() << " s\n";
    std::cout << "Total Words:     " << total_words << "\n";
    std::cout << "Throughput:      " << (total_words / duration.count()) << " words/sec\n";
}