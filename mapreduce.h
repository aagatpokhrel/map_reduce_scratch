#ifndef MAPREDUCE_H
#define MAPREDUCE_H

#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <map>

// A queue with mutex and conditional var to make thread-safe
template <typename T>
class SafeQueue {
private:
    std::queue<T> q;
    std::mutex mtx;
    std::condition_variable cv;
    bool finished = false; //completed, no more addition

public:
    void push(T val) {
        std::lock_guard<std::mutex> lock(mtx);
        q.push(val);
        cv.notify_one();
    }

    // return true if value
    bool pop(T& val) {
        std::unique_lock<std::mutex> lock(mtx);
        while (q.empty() && !finished) {
            cv.wait(lock);
        }
        if (q.empty() && finished) return false;
        val = q.front();
        q.pop();
        return true;
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

// hash to map key to reducers
int partition_key(const std::string& key, int R) {
    std::hash<std::string> hasher;
    return hasher(key) % R;
}

#endif