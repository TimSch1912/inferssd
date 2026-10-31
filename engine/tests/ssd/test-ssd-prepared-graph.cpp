// SPDX-License-Identifier: Apache-2.0
// Exercise real scheduler copies/allocator and CPU math across incompatible
// memory domains. The second domain is host-backed; this is not a GPU gate.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend-impl.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) { if (!ok) { throw std::runtime_error(message); } }

struct separate_domain {
    ggml_backend_buffer_type type = {};
    ggml_backend_device device = {};
    ggml_backend backend = {};
    unsigned allocations = 0, scratch_reserves = 0;
    bool fail_allocation = false, fail_scratch = false;
    ggml_backend_reg reg = {};
    std::vector<std::function<void()>> pending;
    std::vector<unsigned> get_batches;
    unsigned async_gets = 0, async_sets = 0, gets_since_wait = 0, waits = 0;
    unsigned compute_calls = 0, fail_compute_at = 0;

    separate_domain() {
        type.context = this;
        type.device = &device;
        type.iface.get_name = [](ggml_backend_buffer_type_t) { return "separate_domain"; };
        type.iface.get_alignment = [](ggml_backend_buffer_type_t) { return size_t(64); };
        type.iface.is_host = [](ggml_backend_buffer_type_t) { return false; };
        type.iface.alloc_buffer = [](ggml_backend_buffer_type_t type, size_t size) -> ggml_backend_buffer_t {
            auto & domain = *static_cast<separate_domain *>(type->context);
            if (domain.fail_allocation) { return nullptr; }
            void * data = nullptr;
            if (posix_memalign(&data, 64, size ? size : 64)) { return nullptr; }
            ++domain.allocations;
            ggml_backend_buffer_i methods = {};
            methods.get_base = [](ggml_backend_buffer_t buffer) { return buffer->context; };
            methods.free_buffer = [](ggml_backend_buffer_t buffer) { std::free(buffer->context); };
            methods.set_tensor = [](ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t bytes) {
                std::memcpy(static_cast<char *>(tensor->data) + offset, data, bytes);
            };
            methods.get_tensor = [](ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t bytes) {
                std::memcpy(data, static_cast<const char *>(tensor->data) + offset, bytes);
            };
            return ggml_backend_buffer_init(type, methods, data, size);
        };
        device.context = this;
        device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
        device.iface.get_buffer_type = [](ggml_backend_dev_t device) { return &static_cast<separate_domain *>(device->context)->type; };
        device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor * tensor) { return tensor->op != GGML_OP_CUSTOM; };
        device.iface.supports_buft = [](ggml_backend_dev_t device, ggml_backend_buffer_type_t type) {
            return type == &static_cast<separate_domain *>(device->context)->type;
        };
        backend.device = &device;
        backend.context = this;
        backend.iface.get_name = [](ggml_backend_t) { return "separate_domain"; };
        backend.iface.graph_reserve = [](ggml_backend_t backend, ggml_cgraph * const *, int) {
            auto & domain = *static_cast<separate_domain *>(backend->context);
            ++domain.scratch_reserves;
            return !domain.fail_scratch;
        };
        backend.iface.graph_compute = [](ggml_backend_t, ggml_cgraph * graph) {
            auto * context = ggml_init({1 << 20, nullptr, false});
            const auto status = ggml_graph_compute_with_ctx(context, graph, 1);
            ggml_free(context);
            return status;
        };
    }

    void enable_async() {
        // Emulate CUDA's stream on host memory: no transfer or graph runs until
        // synchronize. This exposes missing fences without touching a GPU.
        reg.iface.get_name = [](ggml_backend_reg_t) { return "CUDA"; };
        device.reg = &reg;
        backend.iface.get_tensor_async = [](ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t bytes) {
            auto & domain = *static_cast<separate_domain *>(backend->context);
            ++domain.async_gets;
            ++domain.gets_since_wait;
            domain.pending.emplace_back([=] { std::memcpy(data, static_cast<const char *>(tensor->data) + offset, bytes); });
        };
        backend.iface.set_tensor_async = [](ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t bytes) {
            auto & domain = *static_cast<separate_domain *>(backend->context);
            ++domain.async_sets;
            const auto * begin = static_cast<const char *>(data);
            std::vector<char> snapshot(begin, begin + bytes);
            domain.pending.emplace_back([=] {
                require(std::memcmp(data, snapshot.data(), bytes) == 0, "host allocation overwritten before H2D completion");
                std::memcpy(static_cast<char *>(tensor->data) + offset, data, bytes);
            });
        };
        backend.iface.synchronize = [](ggml_backend_t backend) {
            auto & domain = *static_cast<separate_domain *>(backend->context);
            ++domain.waits;
            if (domain.gets_since_wait) { domain.get_batches.push_back(domain.gets_since_wait); }
            domain.gets_since_wait = 0;
            for (auto & operation : domain.pending) { operation(); }
            domain.pending.clear();
        };
        backend.iface.graph_compute = [](ggml_backend_t backend, ggml_cgraph * graph) {
            auto & domain = *static_cast<separate_domain *>(backend->context);
            if (++domain.compute_calls == domain.fail_compute_at) { return GGML_STATUS_FAILED; }
            domain.pending.emplace_back([=] {
                auto * context = ggml_init({1 << 20, nullptr, false});
                const auto status = ggml_graph_compute_with_ctx(context, graph, 1);
                ggml_free(context);
                require(status == GGML_STATUS_SUCCESS, "queued graph failed");
            });
            return GGML_STATUS_SUCCESS;
        };
    }
};

using scheduler_ptr = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;
struct executor {
    separate_domain domain;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu {ggml_backend_cpu_init(), &ggml_backend_free};
    scheduler_ptr scheduler {nullptr, &ggml_backend_sched_free};
    explicit executor(bool parallel = false) {
        ggml_backend_t backends[] = {&domain.backend, cpu.get()};
        scheduler.reset(ggml_backend_sched_new(backends, nullptr, 2, 128, parallel, false));
        require(bool(cpu) && bool(scheduler), "backend startup failed");
        ggml_backend_sched_set_auto_reserve(scheduler.get(), false);
    }
};

struct graph_case {
    ggml_context * context = ggml_init({1 << 20, nullptr, true});
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_tensor * x = nullptr, * a = nullptr, * b = nullptr, * output = nullptr;
    ggml_cgraph * graph = nullptr;
    explicit graph_case(executor & executor) {
        require(context, "metadata allocation failed");
        x = ggml_new_tensor_1d(context, GGML_TYPE_F32, 16);
        ggml_set_name(x, "input"); ggml_set_input(x);
        auto * weight = ggml_new_tensor_1d(context, GGML_TYPE_F32, 16);
        ggml_set_name(weight, "weight");
        weight_buffer = ggml_backend_buft_alloc_buffer(&executor.domain.type, 64);
        require(weight_buffer, "weight allocation failed");
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_alloc(weight_buffer, weight, ggml_backend_buffer_get_base(weight_buffer));
        const std::array<float, 16> ones {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        ggml_backend_tensor_set(weight, ones.data(), 0, sizeof(ones));
        a = ggml_add(context, weight, x); ggml_set_name(a, "domain_add");
        ggml_tensor * args[] = {a};
        auto twice = +[](ggml_tensor * dst, int ith, int, void *) {
            if (ith) { return; }
            for (int i = 0; i < 16; ++i) { static_cast<float *>(dst->data)[i] = 2 * static_cast<const float *>(dst->src[0]->data)[i]; }
        };
        b = ggml_custom_4d(context, GGML_TYPE_F32, 16, 1, 1, 1, args, 1, twice, 1, nullptr);
        ggml_set_name(b, "cpu_custom");
        output = ggml_add(context, weight, b); ggml_set_name(output, "output"); ggml_set_output(output);
        graph = ggml_new_graph_custom(context, 128, false); ggml_build_forward_expand(graph, output);
        // Assignments make is_reset=false, but do not create split metadata.
        ggml_backend_sched_set_tensor_backend(executor.scheduler.get(), x, executor.cpu.get());
        ggml_backend_sched_set_tensor_backend(executor.scheduler.get(), a, &executor.domain.backend);
        ggml_backend_sched_set_tensor_backend(executor.scheduler.get(), b, executor.cpu.get());
        ggml_backend_sched_set_tensor_backend(executor.scheduler.get(), output, &executor.domain.backend);
    }
    ~graph_case() { ggml_backend_buffer_free(weight_buffer); ggml_free(context); }
    std::array<ggml_tensor *, 3> copies() const { return {a->src[1], b->src[0], output->src[1]}; }
    void compute(executor & executor) {
        std::array<float, 16> input, result;
        input.fill(1);
        ggml_backend_tensor_set(x, input.data(), 0, sizeof(input));
        require(ggml_backend_sched_graph_compute(executor.scheduler.get(), graph) == GGML_STATUS_SUCCESS, "cross-domain compute failed");
        require(ggml_backend_sched_get_n_splits(executor.scheduler.get()) == 3, "graph lost a memory-domain crossing");
        ggml_backend_tensor_get(output, result.data(), 0, sizeof(result));
        for (const float value : result) { require(value == 5, "cross-domain copies lost input data (expected 5)"); }
    }
};

static void numerical(bool parallel) {
    executor executor(parallel);
    // Preparation synchronizes (preserving the scheduler's stable copy-0
    // policy), but all configured parallel input-copy metadata must survive.
    require(ggml_backend_sched_get_n_copies(executor.scheduler.get()) == (parallel ? 4 : 1), "parallel copies were not configured");
    for (int iteration = 0; iteration < 2; ++iteration) {
        ggml_backend_sched_reset(executor.scheduler.get());
        graph_case graph(executor);
        size_t sizes[2] = {};
        const unsigned allocations = executor.domain.allocations;
        require(ggml_backend_sched_prepare_graph(executor.scheduler.get(), graph.graph, sizes), "explicit assignments prevented preparation");
        require(sizes[0] && sizes[1] && executor.domain.allocations == allocations, "probe allocated storage or lost a domain");
        const auto copies = graph.copies();
        require(ggml_backend_sched_reserve_prepared(executor.scheduler.get(), graph.graph), "prepared reservation failed");
        require(copies == graph.copies(), "reservation rewrote copy metadata");
        require(ggml_backend_sched_alloc_prepared(executor.scheduler.get(), graph.graph), "prepared allocation failed");
        require(copies == graph.copies(), "allocation rewrote copy metadata");
        graph.compute(executor);
        ggml_backend_sched_reset(executor.scheduler.get());
    }
    require(executor.domain.scratch_reserves == 2, "backend scratch was not reserved once per graph");
}

static void invalidation_and_failure() {
    for (int mode = 0; mode < 4; ++mode) {
        executor executor;
        graph_case graph(executor);
        size_t sizes[2] = {};
        require(ggml_backend_sched_prepare_graph(executor.scheduler.get(), graph.graph, sizes), "failure case preparation failed");
        if (mode == 0) {
            ggml_backend_sched_reset(executor.scheduler.get());
        } else if (mode == 1) {
            require(!ggml_backend_sched_alloc_prepared(executor.scheduler.get(), graph.graph), "allocation bypassed resource reservation");
        } else {
            executor.domain.fail_allocation = mode == 2;
            executor.domain.fail_scratch = mode == 3;
            require(!ggml_backend_sched_reserve_prepared(executor.scheduler.get(), graph.graph), "injected storage/scratch failure was accepted");
        }
        require(!ggml_backend_sched_reserve_prepared(executor.scheduler.get(), graph.graph), "cancelled transaction remained reservable");
        require(!ggml_backend_sched_alloc_prepared(executor.scheduler.get(), graph.graph), "cancelled transaction remained executable");
    }
}

static void asynchronous_copies(bool parallel = false, bool callback = false, bool fail = false, bool mixed = false) {
    executor executor(parallel);
    executor.domain.enable_async();
    auto * sched = executor.scheduler.get();
    ggml_backend_sched_set_auto_reserve(sched, true);
    auto & domain = executor.domain;
    auto * context = ggml_init({1 << 20, nullptr, true});
    auto * x = ggml_new_tensor_1d(context, GGML_TYPE_F32, 16);
    ggml_set_input(x);
    ggml_backend_sched_set_tensor_backend(sched, x, executor.cpu.get());
    ggml_tensor * value = x;
    auto sum = +[](ggml_tensor * dst, int ith, int, void *) {
        if (ith) { return; }
        for (int i = 0; i < 16; ++i) {
            static_cast<float *>(dst->data)[i] = static_cast<const float *>(dst->src[0]->data)[i] +
                                               static_cast<const float *>(dst->src[1]->data)[i];
        }
    };
    // Repeated crossings let the real allocator recycle earlier CPU results.
    for (int layer = 0; layer < 3; ++layer) {
        auto * a = ggml_scale(context, value, 2);
        auto * b = ggml_scale(context, value, 3);
        ggml_backend_sched_set_tensor_backend(sched, a, &domain.backend);
        ggml_backend_sched_set_tensor_backend(sched, b, &domain.backend);
        // A flagged input makes the whole batch use the legacy path, draining
        // preceding uploads before copies can overwrite recycled host storage.
        if (mixed && layer == 1) { ggml_set_input(b); }
        ggml_tensor * args[] = {a, b};
        value = ggml_custom_4d(context, GGML_TYPE_F32, 16, 1, 1, 1, args, 2, sum, 1, nullptr);
        ggml_backend_sched_set_tensor_backend(sched, value, executor.cpu.get());
    }
    auto * output = ggml_scale(context, value, 2);
    ggml_set_output(output);
    ggml_backend_sched_set_tensor_backend(sched, output, &domain.backend);
    auto * graph = ggml_new_graph_custom(context, 128, false);
    ggml_build_forward_expand(graph, output);
    if (callback) {
        ggml_backend_sched_set_eval_callback(sched, [](ggml_tensor *, bool, void *) { return true; }, nullptr);
    }
    require(ggml_backend_sched_alloc_graph(sched, graph), "async graph allocation failed");
    static const bool requested = [] {
        const char * env = std::getenv("INFERSSD_EXP_ASYNC_COPIES");
        return env && std::strcmp(env, "1") == 0;
    }();
    const bool enabled = requested && !parallel && !callback;
    for (int iteration = 1; iteration <= 2; ++iteration) {
        std::array<float, 16> input, result;
        input.fill(float(iteration));
        ggml_backend_tensor_set(x, input.data(), 0, sizeof(input));
        domain.async_gets = domain.async_sets = domain.gets_since_wait = domain.waits = 0;
        domain.get_batches.clear();
        domain.compute_calls = 0;
        domain.fail_compute_at = fail ? 2 : 0; // Fail after queuing the first CPU result's H2D.
        const auto status = ggml_backend_sched_graph_compute_async(sched, graph);
        require(status == (fail ? GGML_STATUS_FAILED : GGML_STATUS_SUCCESS), "unexpected async copy graph status");
        if (enabled) {
            require(domain.pending.empty(), "async copies escaped graph lifetime guard");
            require(domain.async_sets == (fail ? 1u : 3u), "CPU results did not use async H2D");
            if (!mixed && !fail) {
                require(domain.get_batches == std::vector<unsigned>({2, 2, 2}), "D2H inputs were not batched");
                require(domain.waits == 5, "extra host sync between H2D and CUDA compute");
            }
        } else {
            require(domain.async_gets == 0 && domain.async_sets == 0, "unsupported/disabled case left legacy path");
        }
        ggml_backend_sched_synchronize(sched);
        if (fail) { continue; }
        ggml_backend_tensor_get(output, result.data(), 0, sizeof(result));
        for (float actual : result) { require(actual == float(250 * iteration), "async copy numerics changed"); }
    }
    ggml_backend_sched_reset(sched);
    ggml_free(context);
}

int main() {
    try {
        numerical(false); numerical(true); invalidation_and_failure();
        asynchronous_copies(); asynchronous_copies(true); asynchronous_copies(false, true);
        asynchronous_copies(false, false, false, true);
        asynchronous_copies(false, false, true);
        // Changing the environment after first use must not switch code paths.
        const char * env = std::getenv("INFERSSD_EXP_ASYNC_COPIES");
        setenv("INFERSSD_EXP_ASYNC_COPIES", env && std::strcmp(env, "1") == 0 ? "0" : "1", 1);
        asynchronous_copies();
        std::cout << "prepared graph: scheduler copies, async batches/lifetimes/fallbacks/failure, parallel copies and reservation checks passed\n";
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
