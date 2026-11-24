// FenceWaitDispatcher concurrency and device-lost regressions (runtime/src/vulkan_backend.h):
//   1. Completion callbacks in run() execute outside mutex_, so a callback can re-entrantly call
//      enqueue_fence_callback() and cancel_fence_waits() without deadlocking.
//   2. Cancelled callbacks in cancel() execute outside mutex_, so a callback can re-entrantly enqueue another wait
//      without deadlocking.
//   3. mark_device_lost() wakes a blocked vkWaitSemaphores() so pending waits complete and cancel()/stop() never hang.
//   4. vkGetSemaphoreCounterValue() returning VK_ERROR_DEVICE_LOST drains pending waits and marks device_lost_ instead
//      of spinning forever.
//   5. cancel() blocks until a callback already dispatched for that semaphore has stopped using it.
//   6. A re-entrant cancel() for a later callback in the same ready batch skips that callback.
//   7. A cancel() from another thread during a ready batch drops the semaphore work that has not started, but
//      completes the notification and the event of the cancelled fence, each once, before it returns.
//   8. The same for a re-entrant cancel(), and a notification completed that way can cancel and enqueue waits itself.
//   9. ALL and ANY multi-fence waits are satisfied once, by the right fence, whichever order their fences are released
//      in and however far the batch has got, and never again (with the D3D12 layer's FenceWaiter too, on Windows).
//  10. A cancel() of waits that are still pending runs both kinds, once, before it returns.
//  11. Device loss completes what is pending, and a cancel() during that drain behaves as in 7.
//  12. A cancel() of a semaphore with one callback running and others not started takes those, waits for the one that
//      runs, tells them then, and leaves no use of the semaphore behind.
// After 7 to 9, 11 and 12 nothing of the batch is left in the dispatcher.
// Skips with exit code 77 when no Vulkan 1.3 device is available. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"
#ifdef _WIN32
#include "../shims/d3d12_fence_waiter.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); std::_Exit(1); } \
} while (0)

namespace {

namespace driver {
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkWaitSemaphores real_wait_semaphores = nullptr;
PFN_vkGetSemaphoreCounterValue real_get_counter = nullptr;

std::atomic<bool> in_wait_semaphores{false};
std::atomic<bool> fail_counter_device_lost{false};
std::atomic<std::uint32_t> counter_calls{0};
std::atomic<std::uint32_t> wait_sem_count{0};
std::atomic<bool> hold_after_wait{false};  // keep run() between vkWaitSemaphores and its counter reads
std::atomic<bool> held_after_wait{false};

VKAPI_ATTR VkResult VKAPI_CALL wait_semaphores(VkDevice device, const VkSemaphoreWaitInfo* info, std::uint64_t timeout) {
    wait_sem_count.store(info->semaphoreCount, std::memory_order_release);
    in_wait_semaphores.store(true, std::memory_order_release);
    const VkResult res = real_wait_semaphores(device, info, timeout);
    in_wait_semaphores.store(false, std::memory_order_release);
    while (hold_after_wait.load(std::memory_order_acquire)) {
        held_after_wait.store(true, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    held_after_wait.store(false, std::memory_order_release);
    return res;
}

VKAPI_ATTR VkResult VKAPI_CALL get_semaphore_counter_value(VkDevice device, VkSemaphore sem, std::uint64_t* value) {
    counter_calls.fetch_add(1, std::memory_order_relaxed);
    if (fail_counter_device_lost.load(std::memory_order_acquire)) return VK_ERROR_DEVICE_LOST;
    return real_get_counter(device, sem, value);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    const std::string_view n = name;
    if (n == "vkWaitSemaphores") return reinterpret_cast<PFN_vkVoidFunction>(wait_semaphores);
    if (n == "vkGetSemaphoreCounterValue") return reinterpret_cast<PFN_vkVoidFunction>(get_semaphore_counter_value);
    return real_get_device(device, name);
}

void install(PFN_vkGetDeviceProcAddr device_proc, VkDevice device) {
    real_get_device = device_proc;
    real_wait_semaphores = reinterpret_cast<PFN_vkWaitSemaphores>(device_proc(device, "vkWaitSemaphores"));
    real_get_counter =
        reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(device_proc(device, "vkGetSemaphoreCounterValue"));
}
}  // namespace driver

bool wait_flag(std::mutex& mu, std::condition_variable& cv, const bool& flag, int timeout_ms = 2000) {
    std::unique_lock lk(mu);
    return cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return flag; });
}

bool wait_until_in_vk_wait(int timeout_ms = 2000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!driver::in_wait_semaphores.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// Waits until run() has re-armed vkWaitSemaphores with exactly `count` semaphores (wakeup + N pending waits).
bool wait_until_wait_sem_count(std::uint32_t count, int timeout_ms = 2000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (driver::wait_sem_count.load(std::memory_order_acquire) != count) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

template<class Done>
bool wait_until(Done done, int timeout_ms = 2000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!done()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// The dispatcher's batch in flight, which it keeps private: to see that a cancel() on another thread has got as far as
// taking its waits, and that a batch leaves nothing behind. (An explicit instantiation may name private members.)
using BatchSlots = std::vector<std::shared_ptr<leonida::FenceWaitDispatcher::DispatchSlot>>;
BatchSlots& batch_slots(leonida::FenceWaitDispatcher&);
std::mutex& batch_mutex(leonida::FenceWaitDispatcher&);
leonida::FenceWaitDispatcher& dispatcher(leonida::VulkanContext&);
template<auto Slots, auto Mutex, auto Dispatcher>
struct Reveal {
    friend BatchSlots& batch_slots(leonida::FenceWaitDispatcher& d) { return d.*Slots; }
    friend std::mutex& batch_mutex(leonida::FenceWaitDispatcher& d) { return d.*Mutex; }
    friend leonida::FenceWaitDispatcher& dispatcher(leonida::VulkanContext& c) { return *(c.*Dispatcher); }
};
template struct Reveal<&leonida::FenceWaitDispatcher::dispatching_, &leonida::FenceWaitDispatcher::mutex_,
                       &leonida::VulkanContext::fence_dispatcher_>;

std::size_t cancelled_slots(leonida::VulkanContext& context, VkSemaphore sem) {
    auto& d = dispatcher(context);
    std::lock_guard lock(batch_mutex(d));
    const auto& slots = batch_slots(d);
    return std::size_t(std::count_if(slots.begin(), slots.end(),
                                     [&](const auto& slot) { return slot->semaphore == sem && slot->cancelled; }));
}

// The batch is over: its last callback has been seen, so only the dispatcher's own bookkeeping may still be running.
bool batch_left_nothing(leonida::VulkanContext& context) {
    return wait_until([&] {
        auto& d = dispatcher(context);
        std::lock_guard lock(batch_mutex(d));
        return batch_slots(d).empty();
    });
}

// Makes one batch of everything `fill` enqueues and signals: run() is held after vkWaitSemaphores returned and before
// it reads a counter, so it finds all of it at once, in the order it was enqueued. `parked` gives run() a wait to be
// in; its value is never reached and the caller cancels it.
template<class Fill>
void one_batch(leonida::VulkanContext& context, const leonida::VulkanFence& parked, Fill fill) {
    context.enqueue_fence_callback(parked.semaphore(), 1, [] {});
    driver::hold_after_wait.store(true, std::memory_order_release);
    EXPECT(wait_until([] { return driver::held_after_wait.load(std::memory_order_acquire); }));
    fill();
    driver::hold_after_wait.store(false, std::memory_order_release);
}

// The callback at the front of a batch: the rest of the batch has not started while it is inside run().
class Gate {
public:
    void run() {
        entered_.store(true, std::memory_order_release);
        EXPECT(wait_until([this] { return open_.load(std::memory_order_acquire); }, 10000));
    }
    bool wait_entered() const {
        return wait_until([this] { return entered_.load(std::memory_order_acquire); });
    }
    void open() { open_.store(true, std::memory_order_release); }

private:
    std::atomic<bool> entered_{false}, open_{false};
};

// The waits of one fence as the D3D12 layer registers them: a FenceWaiter's hit, a held CPU Signal (which uses the
// semaphore) and a single-event wait.
std::atomic<int> g_events{0};
struct Waits {
    std::atomic<int> notified{0}, used_semaphore{0};
    void enqueue(leonida::VulkanContext& context, VkSemaphore sem, std::uint64_t value) {
        context.enqueue_fence_notification(sem, value, [this] { ++notified; });
        context.enqueue_fence_callback(sem, value, [this] { ++used_semaphore; });
        context.enqueue_fence_wait(sem, value, this);
    }
};

// A multi-fence wait: counts down as the D3D12 layer's FenceWaiter does (which is Windows-only), and counts how often
// it was satisfied. On Windows a real FenceWaiter counts along. Its event's handle stays open to the end as a
// tripwire: setting it after it was consumed would use a handle that the application may have closed by then, and a
// closed handle would not show that, as SetEvent on one just fails.
class MultiWait {
public:
    explicit MultiWait(unsigned count) : remaining_(count) {
#ifdef _WIN32
        event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        EXPECT(event_);
        waiter_ = std::make_unique<rt::d3d12::FenceWaiter>(event_, count);
#endif
    }
    ~MultiWait() {
#ifdef _WIN32
        CloseHandle(event_);
#endif
    }
    MultiWait(const MultiWait&) = delete;
    MultiWait& operator=(const MultiWait&) = delete;

    void hit() {
        unsigned left = remaining_.load();
        while (left && !remaining_.compare_exchange_weak(left, left - 1)) {
        }
        if (left == 1) ++satisfied_;
#ifdef _WIN32
        waiter_->hit();
#endif
    }

    // How often the wait was satisfied so far. Call it between any two hits: the event is consumed here, so that
    // a second signal shows.
    int satisfied() {
#ifdef _WIN32
        if (WaitForSingleObject(event_, 0) == WAIT_OBJECT_0) ++event_signals_;
        EXPECT(event_signals_ == satisfied_.load());
        EXPECT(waiter_->done() == (remaining_.load() == 0));
#endif
        return satisfied_.load();
    }

private:
    std::atomic<unsigned> remaining_;
    std::atomic<int> satisfied_{0};
#ifdef _WIN32
    HANDLE event_ = nullptr;
    std::unique_ptr<rt::d3d12::FenceWaiter> waiter_;
    int event_signals_ = 0;
#endif
};

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    if (!d.create(false)) {
        std::fprintf(stderr, "vk_fence_dispatcher_sync_test: no Vulkan 1.3 device, skipped\n");
        return 77;
    }
    driver::install(d.device_proc, d.device);

    // 1. Re-entrant completion callback in run(): callback calls enqueue_fence_callback() and cancel_fence_waits().
    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        VulkanFence fence_a(context, 0);
        VulkanFence fence_b(context, 0);
        VulkanFence fence_c(context, 0);
        EXPECT(fence_a.semaphore() != VK_NULL_HANDLE && fence_b.semaphore() != VK_NULL_HANDLE &&
               fence_c.semaphore() != VK_NULL_HANDLE);

        std::mutex mu;
        std::condition_variable cv;
        bool first_ran = false;
        bool chained_ran = false;
        bool c_cancelled = false;

        context.enqueue_fence_callback(fence_c.semaphore(), 99, [&] {
            std::lock_guard lk(mu);
            c_cancelled = true;
            cv.notify_all();
        });
        context.enqueue_fence_callback(fence_a.semaphore(), 1, [&] {
            context.enqueue_fence_callback(fence_b.semaphore(), 1, [&] {
                std::lock_guard lk(mu);
                chained_ran = true;
                cv.notify_all();
            });
            context.cancel_fence_waits(fence_c.semaphore());
            std::lock_guard lk(mu);
            first_ran = true;
            cv.notify_all();
        });

        EXPECT(fence_a.signal(1));
        EXPECT(wait_flag(mu, cv, first_ran));
        EXPECT(wait_flag(mu, cv, c_cancelled));
        EXPECT(fence_b.signal(1));
        EXPECT(wait_flag(mu, cv, chained_ran));

        context.cancel_fence_waits(fence_a.semaphore());
        context.cancel_fence_waits(fence_b.semaphore());
        std::printf("fence dispatcher: re-entrant completion callback in run() ok\n");

        // 2. Re-entrant callback in cancel(): cancelled callback calls enqueue_fence_callback() on another fence.
        VulkanFence fence_d(context, 0);
        VulkanFence fence_e(context, 0);
        EXPECT(fence_d.semaphore() != VK_NULL_HANDLE && fence_e.semaphore() != VK_NULL_HANDLE);

        bool cancel_returned = false;
        bool e_ran = false;
        context.enqueue_fence_callback(fence_d.semaphore(), 50, [&] {
            context.enqueue_fence_callback(fence_e.semaphore(), 1, [&] {
                std::lock_guard lk(mu);
                e_ran = true;
                cv.notify_all();
            });
        });

        std::thread cancel_thread([&] {
            context.cancel_fence_waits(fence_d.semaphore());
            std::lock_guard lk(mu);
            cancel_returned = true;
            cv.notify_all();
        });
        const bool ok_cancel = wait_flag(mu, cv, cancel_returned);
        EXPECT(ok_cancel);
        cancel_thread.join();

        EXPECT(fence_e.signal(1));
        EXPECT(wait_flag(mu, cv, e_ran));
        context.cancel_fence_waits(fence_e.semaphore());
        std::printf("fence dispatcher: re-entrant callback in cancel() ok\n");
    }

    // 3. mark_device_lost() wakes blocked vkWaitSemaphores(), completes pending waits, and stop()/cancel() never hang.
    {
        driver::in_wait_semaphores.store(false, std::memory_order_release);
        std::mutex mu;
        std::condition_variable cv;
        bool callback_fired = false;
        bool teardown_done = false;

        std::thread lost_test([&] {
            VulkanContext lost_ctx(driver::get_device, d.device, true);
            if (!lost_ctx.loaded()) std::_Exit(1);
            VulkanFence stuck(lost_ctx, 0);
            if (stuck.semaphore() == VK_NULL_HANDLE) std::_Exit(1);

            lost_ctx.enqueue_fence_callback(stuck.semaphore(), 100, [&] {
                std::lock_guard lk(mu);
                callback_fired = true;
                cv.notify_all();
            });
            if (!wait_until_in_vk_wait()) std::_Exit(1);

            lost_ctx.mark_device_lost();
            if (!wait_flag(mu, cv, callback_fired)) std::_Exit(1);

            // cancel() and ~VulkanContext() -> stop() must return promptly even after device_lost_ is set.
            lost_ctx.cancel_fence_waits(stuck.semaphore());
        });

        std::thread joiner([&] {
            lost_test.join();
            std::lock_guard lk(mu);
            teardown_done = true;
            cv.notify_all();
        });
        const bool ok_lost = wait_flag(mu, cv, teardown_done, 3000);
        EXPECT(ok_lost);
        joiner.join();
        EXPECT(callback_fired);
        std::printf("fence dispatcher: mark_device_lost() wakes blocked wait and teardown completes ok\n");
    }

    // 4. vkGetSemaphoreCounterValue() returning VK_ERROR_DEVICE_LOST drains pending waits instead of spinning forever.
    {
        driver::in_wait_semaphores.store(false, std::memory_order_release);
        driver::fail_counter_device_lost.store(false, std::memory_order_release);
        driver::counter_calls.store(0, std::memory_order_release);

        std::mutex mu;
        std::condition_variable cv;
        bool drained = false;
        bool done = false;
        bool marked_lost = false;
        std::uint32_t observed_calls = 0;

        std::thread err_test([&] {
            VulkanContext err_ctx(driver::get_device, d.device, true);
            if (!err_ctx.loaded()) std::_Exit(1);
            VulkanFence sem(err_ctx, 0);
            if (sem.semaphore() == VK_NULL_HANDLE) std::_Exit(1);

            err_ctx.enqueue_fence_callback(sem.semaphore(), 1, [&] {
                std::lock_guard lk(mu);
                drained = true;
                cv.notify_all();
            });
            if (!wait_until_in_vk_wait()) std::_Exit(1);

            driver::fail_counter_device_lost.store(true, std::memory_order_release);
            if (!sem.signal(1)) std::_Exit(1);
            if (!wait_flag(mu, cv, drained, 1500)) std::_Exit(1);

            marked_lost = err_ctx.is_device_lost();
            observed_calls = driver::counter_calls.load(std::memory_order_acquire);
            err_ctx.cancel_fence_waits(sem.semaphore());
        });

        std::thread joiner([&] {
            err_test.join();
            std::lock_guard lk(mu);
            done = true;
            cv.notify_all();
        });
        const bool ok_err = wait_flag(mu, cv, done, 3000);
        driver::fail_counter_device_lost.store(false, std::memory_order_release);
        EXPECT(ok_err);
        joiner.join();
        EXPECT(drained);
        EXPECT(marked_lost);
        EXPECT(observed_calls >= 1 && observed_calls <= 4);
        std::printf("fence dispatcher: vkGetSemaphoreCounterValue DEVICE_LOST drains without spin (%u calls) ok\n",
                    observed_calls);
    }

    // 5. cancel() must not return while run() is dispatching a callback that still uses the semaphore.
    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        VulkanFence fence(context, 0);
        const VkSemaphore sem = fence.semaphore();
        EXPECT(sem != VK_NULL_HANDLE);

        std::mutex mu;
        std::condition_variable cv;
        bool entered = false, used = false, release = false, cancelled = false;

        context.enqueue_fence_callback(sem, 1, [&] {
            {
                std::unique_lock lk(mu);
                entered = true;
                cv.notify_all();
                cv.wait(lk, [&] { return release; });
            }
            std::uint64_t value = 0;
            driver::real_get_counter(d.device, sem, &value);
            std::lock_guard lk(mu);
            used = true;
            cv.notify_all();
        });

        EXPECT(fence.signal(1));
        EXPECT(wait_flag(mu, cv, entered));

        std::thread cancel_thread([&] {
            context.cancel_fence_waits(sem);
            std::lock_guard lk(mu);
            cancelled = true;
            cv.notify_all();
        });
        {
            std::unique_lock lk(mu);
            EXPECT(!cv.wait_for(lk, std::chrono::milliseconds(150), [&] { return cancelled; }));
        }
        {
            std::lock_guard lk(mu);
            release = true;
        }
        cv.notify_all();
        EXPECT(wait_flag(mu, cv, cancelled));
        EXPECT(used);
        cancel_thread.join();
        std::printf("fence dispatcher: cancel waits for an in-flight callback ok\n");
    }

    // 6. A re-entrant cancel() for a later callback in the same ready batch must skip that callback instead of
    //    returning while it is still going to run with a semaphore the caller has already destroyed.
    {
        driver::in_wait_semaphores.store(false, std::memory_order_release);
        driver::wait_sem_count.store(0, std::memory_order_release);
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        VulkanFence fence_a(context, 0);
        VulkanFence fence_b(context, 0);
        const VkSemaphore sem_a = fence_a.semaphore();
        const VkSemaphore sem_b = fence_b.semaphore();
        EXPECT(sem_a != VK_NULL_HANDLE && sem_b != VK_NULL_HANDLE && sem_a != sem_b);

        std::mutex mu;
        std::condition_variable cv;
        bool a_done = false, cancel_returned = false, b_after_cancel = false;

        context.enqueue_fence_callback(sem_a, 1, [&] {
            context.cancel_fence_waits(sem_b);
            std::lock_guard lk(mu);
            // B may legitimately run inside cancel() when it was still pending; it must not run after this point.
            cancel_returned = true;
            a_done = true;
            cv.notify_all();
        });
        EXPECT(wait_until_wait_sem_count(2));  // wakeup + A
        context.enqueue_fence_callback(sem_b, 1, [&] {
            std::lock_guard lk(mu);
            if (cancel_returned) b_after_cancel = true;
            cv.notify_all();
        });
        EXPECT(wait_until_wait_sem_count(3));  // wakeup + A + B: one ready batch

        EXPECT(fence_a.signal(1));
        EXPECT(fence_b.signal(1));
        EXPECT(wait_flag(mu, cv, a_done));
        // Give the dispatcher time to walk the rest of the batch.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        EXPECT(!b_after_cancel);

        context.cancel_fence_waits(sem_a);
        context.cancel_fence_waits(sem_b);
        std::printf("fence dispatcher: re-entrant cancel skips later callback in the same batch ok\n");
    }

    // 7. cancel() from another thread while a ready batch is running, then the semaphore is destroyed, as ~Fence does.
    //    The tail at the end of the batch says when all of it has been dispatched, so the counts are final.
    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        g_events.store(0);
        context.set_signal_event_fn([](void*) { ++g_events; });
        VulkanFence parked(context, 0), front(context, 0), last(context, 0);
        std::optional<VulkanFence> released(std::in_place, context, std::uint64_t{0});
        Gate gate;
        Waits waits;
        std::atomic<bool> tail{false};
        one_batch(context, parked, [&] {
            context.enqueue_fence_callback(front.semaphore(), 1, [&] { gate.run(); });
            waits.enqueue(context, released->semaphore(), 1);
            context.enqueue_fence_callback(last.semaphore(), 1, [&] { tail.store(true); });
            EXPECT(front.signal(1) && released->signal(1) && last.signal(1));
        });
        EXPECT(gate.wait_entered());
        context.cancel_fence_waits(released->semaphore());
        // Told when the cancel() that took them returns, not when the batch gets there.
        EXPECT(waits.notified.load() == 1 && g_events.load() == 1);
        context.cancel_fence_waits(released->semaphore());  // and not again
        released.reset();
        EXPECT(waits.used_semaphore.load() == 0 && !tail.load());
        gate.open();
        EXPECT(wait_until([&] { return tail.load(); }));
        EXPECT(waits.used_semaphore.load() == 0);  // the held Signal is still dropped
        EXPECT(waits.notified.load() == 1);        // and the multi-fence wait is not
        EXPECT(g_events.load() == 1);
        EXPECT(batch_left_nothing(context));
        context.cancel_fence_waits(parked.semaphore());
        std::printf("fence dispatcher: cancel during a ready batch completes notifications once ok\n");
    }

    // 8. The same from a callback of the batch, on the dispatcher thread. The completed notification then cancels a
    //    later fence of the batch and enqueues a new wait, neither of which may deadlock or be lost.
    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        g_events.store(0);
        context.set_signal_event_fn([](void*) { ++g_events; });
        VulkanFence parked(context, 0), front(context, 0), later(context, 0), fresh(context, 0), last(context, 0);
        std::optional<VulkanFence> released(std::in_place, context, std::uint64_t{0});
        Waits waits, later_waits;
        std::atomic<int> chained{0}, fresh_ran{0};
        std::atomic<bool> tail{false};
        one_batch(context, parked, [&] {
            context.enqueue_fence_callback(front.semaphore(), 1, [&] {
                context.cancel_fence_waits(released->semaphore());
                released.reset();
            });
            waits.enqueue(context, released->semaphore(), 1);
            context.enqueue_fence_notification(released->semaphore(), 1, [&] {
                ++chained;
                context.cancel_fence_waits(later.semaphore());
                context.enqueue_fence_notification(fresh.semaphore(), 1, [&] { ++fresh_ran; });
            });
            later_waits.enqueue(context, later.semaphore(), 1);
            context.enqueue_fence_callback(last.semaphore(), 1, [&] { tail.store(true); });
            EXPECT(front.signal(1) && released->signal(1) && later.signal(1) && last.signal(1));
        });
        EXPECT(wait_until([&] { return tail.load(); }));
        EXPECT(waits.notified.load() == 1 && waits.used_semaphore.load() == 0);
        EXPECT(chained.load() == 1);
        EXPECT(later_waits.notified.load() == 1 && later_waits.used_semaphore.load() == 0);
        EXPECT(g_events.load() == 2);
        EXPECT(fresh.signal(1) && wait_until([&] { return fresh_ran.load() == 1; }));
        EXPECT(batch_left_nothing(context));
        context.cancel_fence_waits(parked.semaphore());
        std::printf("fence dispatcher: re-entrant cancel during a ready batch completes notifications once ok\n");
    }

    // 9. Multi-fence waits over three fences, released in every order: all of them before the first is dispatched, or
    //    one before each. A gate before every notification stops the dispatcher there, so at most one fence is told
    //    between two looks at the wait. ALL is satisfied by the third fence that is told, released or dispatched, ANY
    //    by the first, and neither by any other.
    for (const bool any : {false, true}) {
        std::array<int, 3> order{0, 1, 2};
        do {
            for (const bool spread : {false, true}) {
                VulkanContext context(driver::get_device, d.device, true);
                EXPECT(context.loaded());
                VulkanFence parked(context, 0), front(context, 0), last(context, 0);
                std::optional<VulkanFence> fences[3];
                for (auto& fence : fences) fence.emplace(context, std::uint64_t{0});
                MultiWait wait(any ? 1u : 3u);
                std::atomic<int> hits[3]{};
                Gate gates[3];
                std::atomic<bool> tail{false};
                one_batch(context, parked, [&] {
                    for (int i = 0; i < 3; ++i) {
                        context.enqueue_fence_callback(front.semaphore(), 1, [&gates, i] { gates[i].run(); });
                        context.enqueue_fence_notification(fences[i]->semaphore(), 1, [&wait, &hits, i] {
                            ++hits[i];
                            wait.hit();
                        });
                    }
                    context.enqueue_fence_callback(last.semaphore(), 1, [&] { tail.store(true); });
                    EXPECT(front.signal(1) && last.signal(1));
                    for (auto& fence : fences) EXPECT(fence->signal(1));
                });
                std::array<bool, 3> told{};
                const auto satisfied_as_told = [&] {
                    const auto count = std::count(told.begin(), told.end(), true);
                    return wait.satisfied() == int(any ? count > 0 : count == 3);
                };
                const auto release = [&](int i) {
                    context.cancel_fence_waits(fences[i]->semaphore());
                    fences[i].reset();
                    told[i] = true;
                    EXPECT(hits[i].load() == 1 && satisfied_as_told());
                };
                for (int step = 0; step < 3; ++step) {
                    EXPECT(gates[step].wait_entered());
                    if (step) told[step - 1] = true;  // dispatched since the last gate, if it was not released before
                    EXPECT(satisfied_as_told());
                    if (spread) release(order[step]);
                    else if (!step) for (const int i : order) release(i);
                    gates[step].open();
                }
                EXPECT(wait_until([&] { return tail.load(); }));
                told[2] = true;
                EXPECT(hits[0].load() == 1 && hits[1].load() == 1 && hits[2].load() == 1);
                EXPECT(satisfied_as_told() && wait.satisfied() == 1);
                EXPECT(batch_left_nothing(context));
                context.cancel_fence_waits(parked.semaphore());
            }
        } while (std::next_permutation(order.begin(), order.end()));
    }
    // One fence cancelled in the batch and one that has not been reached: ALL waits for the latter, which adds nothing
    // to ANY.
    for (const bool any : {false, true}) {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        VulkanFence parked(context, 0), front(context, 0), last(context, 0), unreached(context, 0);
        std::optional<VulkanFence> released(std::in_place, context, std::uint64_t{0});
        MultiWait wait(any ? 1u : 2u);
        std::atomic<int> hits{0};
        const auto hit = [&wait, &hits] {
            wait.hit();
            ++hits;
        };
        Gate gate;
        std::atomic<bool> tail{false};
        one_batch(context, parked, [&] {
            context.enqueue_fence_callback(front.semaphore(), 1, [&] { gate.run(); });
            context.enqueue_fence_notification(released->semaphore(), 1, hit);
            context.enqueue_fence_notification(unreached.semaphore(), 1, hit);
            context.enqueue_fence_callback(last.semaphore(), 1, [&] { tail.store(true); });
            EXPECT(front.signal(1) && released->signal(1) && last.signal(1));
        });
        EXPECT(gate.wait_entered());
        context.cancel_fence_waits(released->semaphore());
        released.reset();
        EXPECT(hits.load() == 1 && wait.satisfied() == int(any));
        gate.open();
        EXPECT(wait_until([&] { return tail.load(); }));
        EXPECT(hits.load() == 1 && wait.satisfied() == int(any));
        EXPECT(unreached.signal(1) && wait_until([&] { return hits.load() == 2; }));
        EXPECT(wait.satisfied() == 1);
        EXPECT(batch_left_nothing(context));
        context.cancel_fence_waits(parked.semaphore());
    }
    std::printf("fence dispatcher: ALL and ANY waits are satisfied once in every release order ok\n");

    // 10. Waits that are still pending: cancel() itself runs all of them before it returns, the held Signal too (its
    //     semaphore is destroyed after cancel(), and it finds its value not reached).
    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        g_events.store(0);
        context.set_signal_event_fn([](void*) { ++g_events; });
        VulkanFence fence(context, 0);
        Waits waits;
        waits.enqueue(context, fence.semaphore(), 5);
        context.cancel_fence_waits(fence.semaphore());
        EXPECT(waits.notified.load() == 1 && waits.used_semaphore.load() == 1 && g_events.load() == 1);
        context.cancel_fence_waits(fence.semaphore());
        EXPECT(fence.signal(5));
        EXPECT(waits.notified.load() == 1 && waits.used_semaphore.load() == 1 && g_events.load() == 1);
        std::printf("fence dispatcher: cancel of pending waits runs each once ok\n");
    }

    // 11. Device loss: the drain of the pending waits is a batch too, whose values were not reached.
    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        g_events.store(0);
        context.set_signal_event_fn([](void*) { ++g_events; });
        VulkanFence parked(context, 0), front(context, 0), kept(context, 0), last(context, 0);
        std::optional<VulkanFence> released(std::in_place, context, std::uint64_t{0});
        Gate gate;
        Waits waits, kept_waits;
        std::atomic<bool> tail{false};
        one_batch(context, parked, [&] {
            context.enqueue_fence_callback(front.semaphore(), 1, [&] { gate.run(); });
            waits.enqueue(context, released->semaphore(), 9);
            kept_waits.enqueue(context, kept.semaphore(), 9);
            context.enqueue_fence_callback(last.semaphore(), 9, [&] { tail.store(true); });
            context.mark_device_lost();
        });
        EXPECT(gate.wait_entered());
        context.cancel_fence_waits(released->semaphore());
        released.reset();
        gate.open();
        EXPECT(wait_until([&] { return tail.load(); }));
        EXPECT(waits.notified.load() == 1 && waits.used_semaphore.load() == 0);
        EXPECT(kept_waits.notified.load() == 1 && kept_waits.used_semaphore.load() == 1);
        EXPECT(g_events.load() == 2);
        Waits after;  // a wait made after the loss completes on the spot
        after.enqueue(context, kept.semaphore(), 9);
        EXPECT(after.notified.load() == 1 && after.used_semaphore.load() == 1 && g_events.load() == 3);
        EXPECT(batch_left_nothing(context));
        std::printf("fence dispatcher: cancel during the device-lost drain completes notifications once ok\n");
    }

    // 12. One callback of a semaphore is running, its other waits have not started. cancel() on another thread takes
    //     those (seen in the batch, so this does not hang on how far that thread got), and has to wait for the one that
    //     runs, which may use the semaphore, before it tells them or returns. Whatever runs may be of either kind.
    for (const bool running_notification : {false, true}) {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        g_events.store(0);
        context.set_signal_event_fn([](void*) { ++g_events; });
        VulkanFence parked(context, 0), last(context, 0);
        std::optional<VulkanFence> released(std::in_place, context, std::uint64_t{0});
        const VkSemaphore sem = released->semaphore();
        Gate gate;
        Waits waits;
        std::atomic<bool> tail{false}, returned{false};
        one_batch(context, parked, [&] {
            if (running_notification) context.enqueue_fence_notification(sem, 1, [&] { gate.run(); });
            else context.enqueue_fence_callback(sem, 1, [&] { gate.run(); });
            waits.enqueue(context, sem, 1);
            context.enqueue_fence_callback(last.semaphore(), 1, [&] { tail.store(true); });
            EXPECT(released->signal(1) && last.signal(1));
        });
        EXPECT(gate.wait_entered());
        std::thread canceller([&] {
            context.cancel_fence_waits(sem);
            returned.store(true);
        });
        EXPECT(wait_until([&] { return cancelled_slots(context, sem) == 3; }));
        for (int i = 0; i < 100; ++i) {
            EXPECT(!returned.load() && waits.notified.load() == 0 && g_events.load() == 0);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        gate.open();
        canceller.join();
        EXPECT(waits.notified.load() == 1 && g_events.load() == 1 && waits.used_semaphore.load() == 0);
        released.reset();
        EXPECT(wait_until([&] { return tail.load(); }));
        EXPECT(waits.notified.load() == 1 && g_events.load() == 1 && waits.used_semaphore.load() == 0);
        EXPECT(batch_left_nothing(context));
        context.cancel_fence_waits(parked.semaphore());
    }
    std::printf("fence dispatcher: cancel waits for the running callback and tells the rest after it ok\n");

    EXPECT(d.validation_errors() == 0);
    std::printf("vk_fence_dispatcher_sync_test: ok\n");
    return 0;
}
