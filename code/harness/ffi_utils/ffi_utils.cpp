/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
namespace py = pybind11;

#include "query_sample_library.h"
#include "loadgen.h"


class QuerySamplesCompletePool
{
public:
    QuerySamplesCompletePool(const size_t numThreads)
        : mStopWork(false)
    {
        for (int i = 0; i < numThreads; ++i)
        {
            mThreads.emplace_back(&QuerySamplesCompletePool::HandleResult, this);
        }
    }

    ~QuerySamplesCompletePool()
    {
        {
            std::unique_lock<std::mutex> lock(mMtx);
            mStopWork = true;
            mCondVar.notify_all();
        }

        for (auto& t : mThreads)
        {
            t.join();
        }
    }

    void Enqueue(uintptr_t ptr, size_t n_resp) noexcept
    {
        std::unique_lock<std::mutex> lock(mMtx);
        mResultQ.emplace_back(std::make_pair(ptr, n_resp));
        mCondVar.notify_one();
    }

    void HandleResult() noexcept
    {
        std::pair<uintptr_t, size_t> res;
        while (true)
        {
            {
                std::unique_lock<std::mutex> lock(mMtx);
                mCondVar.wait(lock, [&]() { return (!mResultQ.empty()) || mStopWork; });

                if (mStopWork)
                {
                    break;
                }

                res = mResultQ.front();
                mResultQ.pop_front();
                mCondVar.notify_one();
            }

            {
                mlperf::QuerySamplesComplete(
                    reinterpret_cast<mlperf::QuerySampleResponse*>(res.first),
                    res.second
                );
            }
        }
    }

private:
    std::vector<std::thread> mThreads;
    std::deque<std::pair<uintptr_t, size_t> > mResultQ;

    std::mutex mMtx;
    std::condition_variable mCondVar;
    bool mStopWork;
};


PYBIND11_MODULE(FFIUtils, m) {
    m.doc() = "A multithreaded pool in C++ for loadgen QuerySamplesComplete";

    py::class_<QuerySamplesCompletePool>(m, "QuerySamplesCompletePool")
        .def(py::init<size_t>())
        .def("enqueue", &QuerySamplesCompletePool::Enqueue);
}
