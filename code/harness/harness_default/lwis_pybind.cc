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

/* Include necessary header files */

// Pybind11
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// Loadgen
#include "loadgen.h"

// TensorRT
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "logger.h"
#include "logging.h"

// LWIS
#include "lwis.hpp"
#include "lwis_buffers.h"

// Google Logging
#include <glog/logging.h>

// General C++
#include <chrono>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <thread>

#include "callback.hpp"
#include "utils.hpp"

// Flags used by harness
#include "harness_flags.hpp"

#include "cuda_profiler_api.h"

namespace py = pybind11;

// add error check
std::function<void(::mlperf::QuerySampleResponse*, std::vector<::mlperf::QuerySampleIndex>&, size_t)> getCallbackMap(
    std::string const& name)
{
    if (callbackMap.find(name) != callbackMap.end())
    {
        return callbackMap[name];
    }
    throw std::invalid_argument("Callback not found");
}

std::shared_ptr<qsl::SampleLibraryEnsemble> createQslPerNumaNode(std::shared_ptr<lwis::Server> sut,
    NumaConfig& numaConfig, std::string& mapPath, std::vector<std::string>& tensorPaths, size_t perfSampleCount,
    size_t padding, bool coalesced, std::vector<bool>& startFromDevice)
{
    std::shared_ptr<qsl::SampleLibraryEnsemble> qsl;

    // Release GIL while doing multi-thread computation
    {
        py::gil_scoped_release release;
        int32_t const nbNumas = numaConfig.size();
        std::vector<qsl::SampleLibraryPtr_t> qsls;
        for (int32_t numaIdx = 0; numaIdx < nbNumas; numaIdx++)
        {
            // Use a thread to construct QSL so that the allocated memory is closer to that NUMA node.
            auto constructQsl = [&]()
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                bindNumaMemPolicy(numaIdx, nbNumas);
                auto oneQsl = std::make_shared<qsl::SampleLibrary>(
                    "LWIS_SampleLibrary", mapPath, tensorPaths, perfSampleCount, padding, coalesced, startFromDevice);
                resetNumaMemPolicy();
                sut->AddSampleLibrary(oneQsl);
                qsls.emplace_back(oneQsl);
            };
            std::thread th(constructQsl);
            bindThreadToCpus(th, numaConfig[numaIdx].second);
            th.join();
        }
        qsl = std::shared_ptr<qsl::SampleLibraryEnsemble>(new qsl::SampleLibraryEnsemble(qsls));
    }

    return qsl;
}

void logDeviceStats(std::shared_ptr<lwis::Server> sut)
{
    // Log device stats
    auto devices = sut->GetDevices();
    for (auto& device : devices)
    {
        auto const& stats = device->GetStats();

        std::cout << "Device " << device->GetName() << " processed:" << std::endl;
        for (auto& elem : stats.m_BatchSizeHistogram)
        {
            std::cout << "  " << elem.second << " batches of size " << elem.first << std::endl;
        }

        std::cout << "  Memcpy Calls: " << stats.m_MemcpyCalls << std::endl;
        std::cout << "  PerSampleCudaMemcpy Calls: " << stats.m_PerSampleCudaMemcpyCalls << std::endl;
        std::cout << "  BatchedCudaMemcpy Calls: " << stats.m_BatchedCudaMemcpyCalls << std::endl;
    }
}

template <typename SampleLibraryType>
void startTest(std::shared_ptr<lwis::Server> server, std::shared_ptr<SampleLibraryType> sampleLibrary,
    mlperf::TestSettings const& testSettings, mlperf::LogSettings const& logSettings)
{
    py::gil_scoped_release release;
    mlperf::StartTest(server.get(), sampleLibrary.get(), testSettings, logSettings);
}

namespace lwis
{
PYBIND11_MODULE(lwis_api, m)
{
    m.doc() = "MLPerf-Inference Python bindings for LWIS harness";

    m.def("parse_numa_config", &parseNumaConfig);
    m.def("get_gpu_to_numa_map", &getGpuToNumaMap);
    m.def("get_callback_map", &getCallbackMap);
    m.def("init_glog", &initGlog);
    m.def("create_qsl_per_numa_node", &createQslPerNumaNode);
    m.def("log_device_stats", &logDeviceStats);

    m.def("start_test", &startTest<qsl::SampleLibrary>);
    m.def("start_test", &startTest<qsl::SampleLibraryEnsemble>);

    m.def("reset", [](std::shared_ptr<Server> sut) { sut.reset(); });
    m.def("reset", [](std::shared_ptr<qsl::SampleLibrary> qsl) { qsl.reset(); });
    m.def("reset", [](std::shared_ptr<qsl::SampleLibraryEnsemble> qsl) { qsl.reset(); });

    py::class_<Server, std::shared_ptr<Server>>(m, "Server")
        .def(py::init<std::string>())
        .def("add_sample_library", &Server::AddSampleLibrary)
        .def("setup", &Server::Setup)
        .def("warmup", &Server::Warmup)
        .def("done", &Server::Done)
        .def("name", &Server::Name)
        .def("issue_query", &Server::IssueQuery)
        .def("flush_queries", &Server::FlushQueries)
        .def("set_response_callback", &Server::SetResponseCallback);

    py::class_<ServerParams>(m, "ServerParams")
        .def(py::init<>())
        .def_readwrite("device_names", &ServerParams::DeviceNames)
        .def_readwrite("engine_names", &ServerParams::EngineNames);

    py::class_<ServerSettings>(m, "ServerSettings")
        .def(py::init<>())
        .def_readwrite("enable_cuda_graphs", &ServerSettings::EnableCudaGraphs)
        .def_readwrite("enable_sync_on_event", &ServerSettings::EnableSyncOnEvent)
        .def_readwrite("enable_spin_wait", &ServerSettings::EnableSpinWait)
        .def_readwrite("enable_device_schedule_spin", &ServerSettings::EnableDeviceScheduleSpin)
        .def_readwrite("enable_dma", &ServerSettings::EnableDma)
        .def_readwrite("enable_dma_staging", &ServerSettings::EnableDmaStaging)
        .def_readwrite("enable_direct_host_access", &ServerSettings::EnableDirectHostAccess)
        .def_readwrite("enable_dla_direct_host_access", &ServerSettings::EnableDLADirectHostAccess)
        .def_readwrite("enable_response", &ServerSettings::EnableResponse)
        .def_readwrite("enable_deque_limit", &ServerSettings::EnableDequeLimit)
        .def_readwrite("enable_batcher_thread_per_device", &ServerSettings::EnableBatcherThreadPerDevice)
        .def_readwrite("enable_cuda_thread_per_device", &ServerSettings::EnableCudaThreadPerDevice)
        .def_readwrite("enable_start_from_device_mem", &ServerSettings::EnableStartFromDeviceMem)
        .def_readwrite("run_infer_on_copy_streams", &ServerSettings::RunInferOnCopyStreams)
        .def_readwrite("use_same_context", &ServerSettings::UseSameContext)
        .def_readwrite("end_on_device", &ServerSettings::EndOnDevice)
        .def_readwrite("use_dla_loop", &ServerSettings::UseDLALoop)
        .def_readwrite("verbose_nvtx", &ServerSettings::VerboseNVTX)
        .def_readwrite("el_ratio", &ServerSettings::elRatio)
        .def_readwrite("gpu_batch_sizes", &ServerSettings::GPUBatchSizes)
        .def_readwrite("gpu_loop_counts", &ServerSettings::GPULoopCounts)
        .def_readwrite("gpu_copy_streams", &ServerSettings::GPUCopyStreams)
        .def_readwrite("gpu_infer_streams", &ServerSettings::GPUInferStreams)
        .def_readwrite("max_gpus", &ServerSettings::MaxGPUs)
        .def_readwrite("dla_batch_sizes", &ServerSettings::DLABatchSizes)
        .def_readwrite("dla_loop_counts", &ServerSettings::DLALoopCounts)
        .def_readwrite("dla_copy_streams", &ServerSettings::DLACopyStreams)
        .def_readwrite("dla_infer_streams", &ServerSettings::DLAInferStreams)
        .def_readwrite("max_dlas", &ServerSettings::MaxDLAs)
        .def_readwrite("force_contiguous", &ServerSettings::ForceContiguous)
        .def_readwrite("complete_threads", &ServerSettings::CompleteThreads)
        .def_readwrite("timeout", &ServerSettings::Timeout)
        .def_readwrite("numa_config", &ServerSettings::m_NumaConfig)
        .def_readwrite("gpu_to_numa_map", &ServerSettings::m_GpuToNumaMap);

    py::class_<qsl::SampleLibrary, std::shared_ptr<qsl::SampleLibrary>>(m, "SampleLibrary")
        .def(py::init<std::string, std::string, std::vector<std::string>, size_t, size_t, bool, std::vector<bool>>(),
            py::arg("name"), py::arg("map_path"), py::arg("tensor_paths"), py::arg("perf_sample_count"),
            py::arg("padding") = 0, py::arg("coalesced") = false,
            py::arg("start_from_device") = std::vector<bool>(1, false))
        .def("name", &qsl::SampleLibrary::Name)
        .def("total_sample_count", &qsl::SampleLibrary::TotalSampleCount)
        .def("performance_sample_count", &qsl::SampleLibrary::PerformanceSampleCount)
        .def("load_samples_to_ram", &qsl::SampleLibrary::LoadSamplesToRam)
        .def("unload_samples_from_ram", &qsl::SampleLibrary::UnloadSamplesFromRam);

    py::class_<qsl::SampleLibraryEnsemble, std::shared_ptr<qsl::SampleLibraryEnsemble>>(m, "SampleLibraryEnsemble")
        .def(py::init<std::vector<qsl::SampleLibraryPtr_t> const&>(), py::arg("qsls"))
        .def("name", &qsl::SampleLibraryEnsemble::Name)
        .def("total_sample_count", &qsl::SampleLibraryEnsemble::TotalSampleCount)
        .def("performance_sample_count", &qsl::SampleLibraryEnsemble::PerformanceSampleCount)
        .def("load_samples_to_ram", &qsl::SampleLibraryEnsemble::LoadSamplesToRam)
        .def("unload_samples_from_ram", &qsl::SampleLibraryEnsemble::UnloadSamplesFromRam);
}
} // namespace lwis
