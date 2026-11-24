// Fence events the host D3D12 runtime (RTX 5070 and WARP) sets or refuses, checked on the Vulkan-backed device and,
// when present, on the host devices too: releasing a fence completes the events still waiting on it (ALL counts it
// as one fence reached, ANY is satisfied), in whichever order three fences are released or signaled, and
// SetEventOnMultipleFenceCompletion with no fences is E_INVALIDARG and sets nothing. On the Vulkan-backed device also
// with the fences released while the layer is dispatching the waits that they reached, stopped in the middle of that.
// No shaders or game data. LEONIDA_VK_VALIDATION=1 fails the test on any validation error.
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "%s line %d: %s\n", who, __LINE__, #condition); return 1; } \
} while (0)

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
};

struct Event {
    HANDLE h = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { if (h) CloseHandle(h); }
    bool set_within(DWORD ms) const { return WaitForSingleObject(h, ms) == WAIT_OBJECT_0; }
};

// Stops the layer's fence dispatcher where the test wants it: after a vkWaitSemaphores, before it looks at what was
// reached, and inside the held CPU Signal that is then the first wait it dispatches. Calls of the test's own thread
// pass. So which waits are dispatched together, and where in them a fence is released, does not depend on timing or
// on the number of CPUs.
namespace driver {
constexpr std::uint64_t kHeldSignal = 777;  // the value of that CPU Signal
PFN_vkGetInstanceProcAddr real_instance = nullptr;
PFN_vkGetDeviceProcAddr real_device = nullptr;
PFN_vkWaitSemaphores real_wait = nullptr;
PFN_vkSignalSemaphore real_signal = nullptr;
DWORD test_thread = 0;

struct Stop {
    std::atomic<bool> closed{false}, reached{false};
    void here() {
        while (closed.load()) {
            reached.store(true);
            Sleep(1);
        }
        reached.store(false);
    }
    bool wait_reached() const {
        for (int i = 0; i < 5000 && !reached.load(); ++i) Sleep(1);
        return reached.load();
    }
};
Stop after_wait, in_held_signal;

// Opens both stops on the way out, also of a check that failed, so that releasing the device never waits for a
// dispatcher that is stopped. Declare it after what it protects.
struct OpenStops {
    ~OpenStops() {
        after_wait.closed.store(false);
        in_held_signal.closed.store(false);
    }
};

VKAPI_ATTR VkResult VKAPI_CALL wait_semaphores(VkDevice device, const VkSemaphoreWaitInfo* info,
                                               std::uint64_t timeout) {
    const VkResult result = real_wait(device, info, timeout);
    if (GetCurrentThreadId() != test_thread) after_wait.here();
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL signal_semaphore(VkDevice device, const VkSemaphoreSignalInfo* info) {
    if (GetCurrentThreadId() != test_thread && info->value == kHeldSignal) in_held_signal.here();
    return real_signal(device, info);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkWaitSemaphores") == 0) return reinterpret_cast<PFN_vkVoidFunction>(wait_semaphores);
    if (std::strcmp(name, "vkSignalSemaphore") == 0) return reinterpret_cast<PFN_vkVoidFunction>(signal_semaphore);
    return real_device(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance(VkInstance instance, const char* name) {
    if (std::strcmp(name, "vkGetDeviceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(get_device);
    return real_instance(instance, name);
}

bool install(const vktest::Device& vk) {
    test_thread = GetCurrentThreadId();
    real_instance = vk.instance_proc;
    real_device = reinterpret_cast<PFN_vkGetDeviceProcAddr>(vk.instance_proc(vk.instance, "vkGetDeviceProcAddr"));
    if (!real_device) return false;
    real_wait = reinterpret_cast<PFN_vkWaitSemaphores>(real_device(vk.device, "vkWaitSemaphores"));
    real_signal = reinterpret_cast<PFN_vkSignalSemaphore>(real_device(vk.device, "vkSignalSemaphore"));
    return real_wait && real_signal;
}
}  // namespace driver

// Three fences, each reached by a Signal or by its release, one after the other in every order: ALL's event is set by
// the third, ANY's by the first, and by no other.
int check_orders(ID3D12Device* device, ID3D12Device1* device1, const char* who) {
    for (const auto flags : {D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY}) {
        const bool any = flags == D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY;
        std::array<int, 3> order{0, 1, 2};
        do {
            for (const unsigned signaled : {0u, 2u, 5u}) {  // bit `turn`: that one is signaled instead of released
                Event event;
                EXPECT(event.h);
                Ref<ID3D12Fence> fences[3];
                for (auto& fence : fences)
                    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
                ID3D12Fence* waited[3] = {fences[0].p, fences[1].p, fences[2].p};
                const UINT64 values[3] = {1, 2, 3};
                EXPECT(SUCCEEDED(device1->SetEventOnMultipleFenceCompletion(waited, values, 3, flags, event.h)));
                EXPECT(!event.set_within(5));
                for (int turn = 0; turn < 3; ++turn) {
                    const int i = order[turn];
                    if (signaled & (1u << turn)) EXPECT(SUCCEEDED(fences[i]->Signal(values[i])));
                    else fences[i].reset();
                    if (turn == (any ? 0 : 2)) EXPECT(event.set_within(2000));
                    else EXPECT(!event.set_within(5));
                }
                EXPECT(!event.set_within(5));
            }
        } while (std::next_permutation(order.begin(), order.end()));
    }
    return 0;
}

// The Vulkan-backed device only. Three fences are reached together and their waits are one batch of the fence
// dispatcher, behind a held CPU Signal, in which it is stopped. `released` of them are released there, in every order.
// No wait is lost: what a release completes is set when it returns, the rest when the dispatcher goes on, each once.
int check_released_in_batch(ID3D12Device* device, ID3D12Device1* device1, const char* who) {
    Ref<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    for (const auto flags : {D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY}) {
        const bool any = flags == D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY;
        std::array<int, 3> order{0, 1, 2};
        do {
            for (int released = 1; released <= 3; ++released) {
                Event event, single[3], done;  // closed after the fences that name them are released
                EXPECT(event.h && single[0].h && single[1].h && single[2].h && done.h);
                Ref<ID3D12Fence> gate, front, fences[3], last;
                for (Ref<ID3D12Fence>* fence : {&gate, &front, &fences[0], &fences[1], &fences[2], &last})
                    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence->p))));
                driver::OpenStops open_stops;

                // The first wait: a CPU Signal held behind a queue Signal, which is held behind `gate`.
                EXPECT(SUCCEEDED(queue->Wait(gate.p, 1)) && SUCCEEDED(queue->Signal(front.p, 5)));
                EXPECT(SUCCEEDED(front->Signal(driver::kHeldSignal)));
                // Then, with the dispatcher stopped, the waits that are to be found with it, and all of them reached.
                driver::after_wait.closed.store(true);
                ID3D12Fence* waited[3] = {fences[0].p, fences[1].p, fences[2].p};
                const UINT64 values[3] = {1, 1, 1};
                EXPECT(SUCCEEDED(device1->SetEventOnMultipleFenceCompletion(waited, values, 3, flags, event.h)));
                for (int i = 0; i < 3; ++i) EXPECT(SUCCEEDED(fences[i]->SetEventOnCompletion(1, single[i].h)));
                EXPECT(SUCCEEDED(last->SetEventOnCompletion(1, done.h)));
                EXPECT(driver::after_wait.wait_reached());
                for (auto& fence : fences) EXPECT(SUCCEEDED(fence->Signal(1)));
                EXPECT(SUCCEEDED(last->Signal(1)) && SUCCEEDED(gate->Signal(1)));
                for (int i = 0; i < 5000 && front->GetCompletedValue() < 5; ++i) Sleep(1);
                EXPECT(front->GetCompletedValue() == 5);
                driver::in_held_signal.closed.store(true);
                driver::after_wait.closed.store(false);
                EXPECT(driver::in_held_signal.wait_reached());

                // The dispatcher is inside the first wait of the batch; none of the others has started.
                EXPECT(!event.set_within(0) && !done.set_within(0));
                int event_set = 0, single_set[3]{};
                bool at_release[3]{}, single_at_release[3]{};
                for (int turn = 0; turn < released; ++turn) {
                    const int i = order[turn];
                    fences[i].reset();
                    at_release[turn] = event.set_within(0);
                    single_at_release[turn] = single[i].set_within(0);
                    event_set += at_release[turn];
                    single_set[i] += single_at_release[turn];
                }
                driver::in_held_signal.closed.store(false);
                EXPECT(done.set_within(5000));  // the last wait of the batch: the others have been dispatched
                event_set += event.set_within(0);
                for (int i = 0; i < 3; ++i) single_set[i] += single[i].set_within(0);
                EXPECT(event_set == 1);  // 0: the multi-fence wait was lost
                EXPECT(single_set[0] == 1 && single_set[1] == 1 && single_set[2] == 1);
                for (int turn = 0; turn < released; ++turn) {
                    EXPECT(at_release[turn] == (turn == (any ? 0 : 2)));
                    EXPECT(single_at_release[turn]);
                }
                // The held CPU Signal, whose fence was not released, has been applied.
                for (int i = 0; i < 5000 && front->GetCompletedValue() != driver::kHeldSignal; ++i) Sleep(1);
                EXPECT(front->GetCompletedValue() == driver::kHeldSignal);
            }
        } while (std::next_permutation(order.begin(), order.end()));
    }
    std::printf("%s: no wait of a fence released in a batch is lost, in any order\n", who);
    return 0;
}

int check(ID3D12Device* device, const char* who) {
    Ref<ID3D12Device1> device1;
    EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device1.p))));
    constexpr auto ALL = D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, ANY = D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY;
    {  // A fence released before its value is reached sets the event.
        ID3D12Fence* fence = nullptr;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))));
        Event event;
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(5, event.h)));
        fence->Release();
        EXPECT(event.set_within(2000));
    }
    for (const auto flags : {ALL, ANY}) {  // Released with a multi-fence wait pending: one fence counted as reached.
        ID3D12Fence* released = nullptr;
        Ref<ID3D12Fence> other;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&released))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&other.p))));
        ID3D12Fence* fences[2] = {released, other.p};
        const UINT64 values[2] = {5, 1};
        Event event;
        EXPECT(SUCCEEDED(device1->SetEventOnMultipleFenceCompletion(fences, values, 2, flags, event.h)));
        released->Release();
        if (flags == ANY) {
            EXPECT(event.set_within(2000));
        } else {
            EXPECT(!event.set_within(100));  // still waiting for the other fence
            EXPECT(SUCCEEDED(other->Signal(1)));
            EXPECT(event.set_within(2000));
        }
    }
    if (check_orders(device, device1.p, who)) return 1;
    {  // No fences: refused, with or without an event, and the event is left alone.
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        ID3D12Fence* fences[1] = {fence.p};
        const UINT64 values[1] = {1};
        Event event;
        for (const auto flags : {ALL, ANY}) {
            EXPECT(device1->SetEventOnMultipleFenceCompletion(fences, values, 0, flags, event.h) == E_INVALIDARG);
            EXPECT(device1->SetEventOnMultipleFenceCompletion(nullptr, nullptr, 0, flags, event.h) == E_INVALIDARG);
            EXPECT(device1->SetEventOnMultipleFenceCompletion(fences, values, 0, flags, nullptr) == E_INVALIDARG);
        }
        EXPECT(!event.set_within(0));
    }
    std::printf("%s: fence events match the host runtime\n", who);
    return 0;
}
}  // namespace

int main() {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready || !driver::install(vk)) return 1;
    int failures = 0;
    {
        Ref<ID3D12Device> ours;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = driver::get_instance;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
        failures += check(ours.p, "vulkan");
        Ref<ID3D12Device1> ours1;
        if (FAILED(ours->QueryInterface(IID_PPV_ARGS(&ours1.p)))) return 1;
        failures += check_released_in_batch(ours.p, ours1.p, "vulkan");
        Ref<IDXGIFactory4> factory;
        if (!hostwine::host_is_wine() && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
            for (const bool warp : {false, true}) {  // the reference the expectations were measured on
                Ref<IDXGIAdapter> adapter;
                Ref<ID3D12Device> host;
                std::string adapter_name;
                if (!hostadapter::reference(factory.p, vk, warp, &adapter.p, adapter_name)) continue;
                if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
                failures += check(host.p, ("host " + adapter_name).c_str());
            }
        }
    }
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", vk.validation_errors());
        return 1;
    }
    return failures ? 1 : 0;
}
