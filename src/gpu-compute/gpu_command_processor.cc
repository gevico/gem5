/*
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "gpu-compute/gpu_command_processor.hh"

#include <algorithm>
#include <cassert>

#include "arch/amdgpu/vega/pagetable_walker.hh"
#include "base/chunk_generator.hh"
#include "debug/GPUCommandProc.hh"
#include "debug/GPUDisp.hh"
#include "debug/GPUInitAbi.hh"
#include "debug/GPUKernelInfo.hh"
#include "dev/amdgpu/amdgpu_device.hh"
#include "dev/amdgpu/interrupt_handler.hh"
#include "gpu-compute/compute_unit.hh"
#include "gpu-compute/dispatcher.hh"
#include "gpu-compute/shader.hh"
#include "mem/abstract_mem.hh"
#include "mem/packet_access.hh"
#include "mem/se_translating_port_proxy.hh"
#include "mem/translating_port_proxy.hh"
#include "params/GPUCommandProcessor.hh"
#include "sim/eventq.hh"
#include "sim/full_system.hh"
#include "sim/process.hh"
#include "sim/proxy_ptr.hh"
#include "sim/sim_exit.hh"
#include "sim/syscall_emul_buf.hh"

namespace gem5
{

GPUCommandProcessor::GPUCommandProcessor(const Params &p)
    : DmaVirtDevice(p), dispatcher(*p.dispatcher), _driver(nullptr),
      walker(p.walker), hsaPP(p.hsapp),
      target_non_blit_kernel_id(p.target_non_blit_kernel_id)
{
    assert(hsaPP);
    hsaPP->setDevice(this);
    dispatcher.setCommandProcessor(this);
}

HSAPacketProcessor&
GPUCommandProcessor::hsaPacketProc()
{
    return *hsaPP;
}

/**
 * Forward the VRAM requestor ID needed for device memory from GPU device.
 */
RequestorID
GPUCommandProcessor::vramRequestorId()
{
    return gpuDevice->vramRequestorId();
}

TranslationGenPtr
GPUCommandProcessor::translate(Addr vaddr, Addr size)
{
    if (!FullSystem) {
        // Grab the process and try to translate the virtual address with it;
        // with new extensions, it will likely be wrong to just arbitrarily
        // grab context zero.
        auto process = sys->threads[0]->getProcessPtr();

        return process->pTable->translateRange(vaddr, size);
    }

    fatal_if(!currentDMAVMID,
             "GPUCommandProcessor DMA requires a valid VMID\n");

    // In full system use the page tables setup by the kernel driver rather
    // than the CPU page tables.
    return TranslationGenPtr(new AMDGPUVM::UserTranslationGen(
        &gpuDevice->getVM(), walker, currentDMAVMID, vaddr, size));
}

void
GPUCommandProcessor::dmaReadVirtForVMID(Addr host_addr, unsigned size,
                                        DmaCallback *cb, void *data,
                                        uint16_t vmid, Tick delay)
{
    uint16_t prev_vmid = currentDMAVMID;
    currentDMAVMID = vmid;
    dmaReadVirt(host_addr, size, cb, data, delay);
    currentDMAVMID = prev_vmid;
}

void
GPUCommandProcessor::dmaWriteVirtForVMID(Addr host_addr, unsigned size,
                                         DmaCallback *cb, void *data,
                                         uint16_t vmid, Tick delay)
{
    uint16_t prev_vmid = currentDMAVMID;
    currentDMAVMID = vmid;
    dmaWriteVirt(host_addr, size, cb, data, delay);
    currentDMAVMID = prev_vmid;
}

void
GPUCommandProcessor::performTimingRead(PacketPtr pkt, int dispType)
{
    // Use the shader to access the CUs and call the read request from
    // the SQC port. Call submit kernel dispatch in the timing response
    // function in receive timing response of SQC port. Schedule this
    // timing read when...just currTick
    ComputeUnit *cu = shader()->cuList[0];
    pkt->senderState = new ComputeUnit::SQCPort::SenderState(
            cu->wfList[0][0], true);
    ComputeUnit::SQCPort::SenderState *sender_state =
        safe_cast<ComputeUnit::SQCPort::SenderState*>(pkt->senderState);
    sender_state->dispatchType = dispType;
    ComputeUnit::SQCPort &sqc_port = cu->sqcPort;

    if (!sqc_port.sendTimingReq(pkt)) {
        sqc_port.retries.push_back(
            std::pair<PacketPtr, Wavefront*>(pkt, sender_state->wavefront)
        );
    }
}

void
GPUCommandProcessor::completeTimingRead(PacketPtr pkt, int dispType)
{
    auto it = kernelDispatchReads.find(pkt);
    panic_if(it == kernelDispatchReads.end(),
             "Completed unknown kernel dispatch timing read pkt=%p", pkt);

    KernelDispatchReadContext *ctx = it->second;
    kernelDispatchReads.erase(it);
    delete pkt;

    panic_if(ctx->dispatchType != dispType,
             "Mismatched kernel dispatch timing read type %d != %d",
             ctx->dispatchType, dispType);

    if (--ctx->pendingReads != 0) {
        return;
    }

    struct KernelDispatchData dispatchData = ctx->dispatchData;
    delete ctx;

    switch (dispType) {
        case ComputeUnit::SQCPort::SenderState::DISPATCH_KERNEL_OBJECT:
            dispatchKernelObject(
                dispatchData.akc, dispatchData.raw_pkt, dispatchData.queue_id,
                dispatchData.host_pkt_addr, dispatchData.vmid);
            break;
        case ComputeUnit::SQCPort::SenderState::DISPATCH_PRELOAD_ARG:
            initPreload(dispatchData.akc, dispatchData.task);
            break;
        default:
            panic("Unknown kernel dispatch timing read type %d", dispType);
    }
}

/**
 * submitDispatchPkt() is the entry point into the CP from the HSAPP
 * and is only meant to be used with AQL kernel dispatch packets.
 * After the HSAPP receives and extracts an AQL packet, it sends
 * it to the CP, which is responsible for gathering all relevant
 * information about a task, initializing CU state, and sending
 * it to the dispatcher for WG creation and dispatch.
 *
 * First we need capture all information from the the AQL pkt and
 * the code object, then store it in an HSAQueueEntry. Once the
 * packet and code are extracted, we extract information from the
 * queue descriptor that the CP needs to perform state initialization
 * on the CU. Finally we call dispatch() to send the task to the
 * dispatcher. When the task completely finishes, we call finishPkt()
 * on the HSA packet processor in order to remove the packet from the
 * queue, and notify the runtime that the task has completed.
 */
void
GPUCommandProcessor::submitDispatchPkt(void *raw_pkt, uint32_t queue_id,
                                       Addr host_pkt_addr)
{
    _hsa_dispatch_packet_t *disp_pkt = (_hsa_dispatch_packet_t*)raw_pkt;
    // The kernel object should be aligned to a 64B boundary, but not
    // necessarily a cache line boundary.
    unsigned akc_alignment_granularity = 64;
    assert(!(disp_pkt->kernel_object & (akc_alignment_granularity - 1)));

    /**
     * Make sure there is not a race condition with invalidates in the L2
     * cache. The full system driver may write directly to memory using
     * large BAR while the L2 cache is allowed to keep data in the valid
     * state between kernel launches. This is a rare event but is required
     * for correctness.
     */
    if (shader()->getNumOutstandingInvL2s() > 0) {
        DPRINTF(GPUCommandProc,
                "Deferring kernel launch due to outstanding L2 invalidates\n");
        shader()->addDeferredDispatch(raw_pkt, queue_id, host_pkt_addr);

        return;
    }

    /**
     * Need to use a raw pointer for DmaVirtDevice API. This is deleted
     * in the dispatchKernelObject method.
     */
    AMDKernelCode *akc = new AMDKernelCode;

    /**
     * The kernel_object is a pointer to the machine code, whose entry
     * point is an 'amd_kernel_code_t' type, which is included in the
     * kernel binary, and describes various aspects of the kernel. The
     * desired entry is the 'kernel_code_entry_byte_offset' field,
     * which provides the byte offset (positive or negative) from the
     * address of the amd_kernel_code_t to the start of the machine
     * instructions.
     *
     * For SE mode we can read from the port proxy. In FS mode, we may need
     * to wait for the guest OS to setup translations, especially when using
     * the KVM CPU, so it is preferred to read the code object using a timing
     * DMA request.
     */
    if (!FullSystem) {
        /**
         * we need to read a pointer in the application's address
         * space to pull out the kernel code descriptor.
         */
        auto *tc = sys->threads[0];
        SETranslatingPortProxy virt_proxy(tc);

        DPRINTF(GPUCommandProc, "reading kernel_object using proxy\n");
        virt_proxy.readBlob(disp_pkt->kernel_object, (uint8_t*)akc,
            sizeof(AMDKernelCode));

        dispatchKernelObject(akc, raw_pkt, queue_id, host_pkt_addr, 0);
    } else {
        /**
         * In full system mode, the page table entry may point to a system
         * page or a device page. System pages use the proxy as normal, but
         * a device page needs to be read from device memory. Check what type
         * it is here.
         */
        bool is_system_page = true;
        Addr phys_addr = disp_pkt->kernel_object;

        uint16_t vmid = hsaPacketProc().getQueueDesc(queue_id)->vmid;
        unsigned tmp_bytes;
        walker->startFunctional(gpuDevice->getVM().getPageTableBase(vmid),
                                phys_addr, tmp_bytes, BaseMMU::Mode::Read,
                                is_system_page);

        DPRINTF(GPUCommandProc, "kernel_object vaddr %#lx paddr %#lx size %d"
                " s:%d\n", disp_pkt->kernel_object, phys_addr,
                sizeof(AMDKernelCode), is_system_page);

        /**
         * System objects use DMA device. Device objects need to use device
         * memory.
         */
        if (is_system_page) {
            DPRINTF(GPUCommandProc,
                    "sending system DMA read for kernel_object\n");

            auto dma_callback =
                new DmaVirtCallback<uint32_t>([=](const uint32_t &) {
                    dispatchKernelObject(akc, raw_pkt, queue_id, host_pkt_addr,
                                         vmid);
                });

            dmaReadVirtForVMID(disp_pkt->kernel_object, sizeof(AMDKernelCode),
                               dma_callback, (void *)akc, vmid);
        } else {
            DPRINTF(GPUCommandProc,
                    "kernel_object in device, using device mem\n");

            auto *dispatchReadCtx = new KernelDispatchReadContext;
            dispatchReadCtx->dispatchData.akc = akc;
            dispatchReadCtx->dispatchData.raw_pkt = raw_pkt;
            dispatchReadCtx->dispatchData.queue_id = queue_id;
            dispatchReadCtx->dispatchData.host_pkt_addr = host_pkt_addr;
            dispatchReadCtx->dispatchData.vmid = vmid;
            dispatchReadCtx->dispatchType =
                ComputeUnit::SQCPort::SenderState::DISPATCH_KERNEL_OBJECT;

            // Read from GPU memory manager one cache line at a time to prevent
            // rare cases where the AKC spans two memory pages.
            ChunkGenerator gen(disp_pkt->kernel_object, sizeof(AMDKernelCode),
                               akc_alignment_granularity);
            for (; !gen.done(); gen.next()) {
                Addr chunk_addr = gen.addr();
                unsigned dummy;
                walker->startFunctional(
                    gpuDevice->getVM().getPageTableBase(vmid), chunk_addr,
                    dummy, BaseMMU::Mode::Read, is_system_page);

                Request::Flags flags = Request::PHYSICAL;
                RequestPtr request = std::make_shared<Request>(chunk_addr,
                    akc_alignment_granularity, flags,
                    walker->getDevRequestor());
                PacketPtr readPkt = new Packet(request, MemCmd::ReadReq);
                readPkt->dataStatic((uint8_t *)akc + gen.complete());
                // If the request spans two device memories, the device memory
                // returned will be null.
                assert(system()->getDeviceMemory(readPkt) != nullptr);
                ++dispatchReadCtx->pendingReads;
                kernelDispatchReads[readPkt] = dispatchReadCtx;
                performTimingRead(readPkt,
                    ComputeUnit::SQCPort::SenderState::DISPATCH_KERNEL_OBJECT);
            }
        }
    }
}

void
GPUCommandProcessor::dispatchKernelObject(AMDKernelCode *akc, void *raw_pkt,
                                          uint32_t queue_id,
                                          Addr host_pkt_addr, uint16_t vmid)
{
    _hsa_dispatch_packet_t *disp_pkt = (_hsa_dispatch_packet_t*)raw_pkt;

    /**
     * If the kernarg_preload_spec_length is non-zero, the CP firmware will
     * append additional bytes to the kernel_code_entry_byte_offset.
     *
     * https://llvm.org/docs/AMDGPUUsage.html#amdgpu-amdhsa-kernarg-preload
     */
    if (akc->kernarg_preload_spec_length != 0) {
        akc->kernel_code_entry_byte_offset += KernargPreloadPktSize;
    }

    sanityCheckAKC(akc);

    DPRINTF(GPUCommandProc, "GPU machine code is %lli bytes from start of the "
        "kernel object\n", akc->kernel_code_entry_byte_offset);

    Addr machine_code_addr = (Addr)disp_pkt->kernel_object
        + akc->kernel_code_entry_byte_offset;

    DPRINTF(GPUCommandProc, "Machine code starts at addr: %#x\n",
        machine_code_addr);

    std::string kernel_name;

    /**
     * BLIT kernels don't have symbol names.  BLIT kernels are built-in compute
     * kernels issued by ROCm to handle DMAs for dGPUs when the SDMA
     * hardware engines are unavailable or explicitly disabled.  They can also
     * be used to do copies that ROCm things would be better performed
     * by the shader than the SDMA engines.  They are also sometimes used on
     * APUs to implement asynchronous memcopy operations from 2 pointers in
     * host memory.  I have no idea what BLIT stands for.
     * */
    bool is_blit_kernel;
    if (!disp_pkt->completion_signal) {
        kernel_name = "Some kernel";
        is_blit_kernel = false;
    } else {
        kernel_name = "Blit kernel";
        is_blit_kernel = true;
    }

    DPRINTF(GPUKernelInfo, "Kernel name: %s\n", kernel_name.c_str());

    GfxVersion gfxVersion = FullSystem ? gpuDevice->getGfxVersion()
                          : driver()->getGfxVersion();
    HSAQueueEntry *task =
        new HSAQueueEntry(kernel_name, queue_id, dynamic_task_id, raw_pkt, akc,
                          host_pkt_addr, machine_code_addr, gfxVersion, vmid);

    if (FullSystem && task->kernargAddr() != 0 && task->kernargSize() > 0) {
        Addr va = task->kernargAddr();
        Addr remaining = task->kernargSize();

        while (remaining > 0) {
            Addr pa = va;
            bool system = true;
            unsigned log_bytes = 0;
            // Use the queue VMID for kernarg translations so dispatch metadata
            // follows the same address space as the AQL queue.
            Fault fault = walker->startFunctional(
                gpuDevice->getVM().getPageTableBase(vmid), pa, log_bytes,
                BaseMMU::Mode::Read, system);

            fatal_if(fault != NoFault,
                     "Failed to translate kernarg range %#x size %#x\n", va,
                     remaining);

            const Addr page_size = Addr(1) << log_bytes;
            const Addr page_offset = va & (page_size - 1);
            const Addr chunk_size =
                std::min(remaining, page_size - page_offset);

            task->addKernargPhysRange(pa, chunk_size, system);
            va += chunk_size;
            remaining -= chunk_size;
        }
    }

    // The driver expects the start time to be in ns
    Tick start_ts = curTick() / sim_clock::as_int::ns;
    dispatchStartTime.insert({disp_pkt->completion_signal, start_ts});
    dispatchVMID.insert({disp_pkt->completion_signal, vmid});

    // Potentially skip a non-blit kernel
    if (!is_blit_kernel && (non_blit_kernel_id < target_non_blit_kernel_id)) {
        DPRINTF(GPUCommandProc, "Skipping non-blit kernel %i (Task ID: %i)\n",
                non_blit_kernel_id, dynamic_task_id);

        auto *disp_pkt_ptr = task->dispPktPtr();
        auto queue_id = task->queueId();
        auto finish_skip = [=] {
            hsaPacketProc().finishPkt(disp_pkt_ptr, queue_id);
            // Notify the run script that a kernel has been skipped
            exitSimLoop("Skipping GPU Kernel");
        };

        if (task->completionSignal()) {
            DPRINTF(GPUDisp, "HSA AQL Kernel Complete with completion "
                    "signal! Addr: %d\n", task->completionSignal());

            auto done = new EventFunctionWrapper(finish_skip, name(), true);
            sendCompletionSignal(task->completionSignal(), vmid, done);
        } else {
            DPRINTF(GPUDisp, "HSA AQL Kernel Complete! No completion "
                             "signal\n");
            finish_skip();
        }

        ++dynamic_task_id;
        ++non_blit_kernel_id;

        delete akc;

        return;
    }

    DPRINTF(GPUCommandProc, "Task ID: %i Got AQL: wg size (%dx%dx%d), "
        "grid size (%dx%dx%d) kernarg addr: %#x, completion "
        "signal addr:%#x\n", dynamic_task_id, disp_pkt->workgroup_size_x,
        disp_pkt->workgroup_size_y, disp_pkt->workgroup_size_z,
        disp_pkt->grid_size_x, disp_pkt->grid_size_y,
        disp_pkt->grid_size_z, disp_pkt->kernarg_address,
        disp_pkt->completion_signal);

    DPRINTF(GPUCommandProc, "Extracted code object: %s (num vector regs: %d, "
        "num scalar regs: %d, code addr: %#x, kernarg size: %d, "
        "LDS size: %d)\n", kernel_name, task->numVectorRegs(),
        task->numScalarRegs(), task->codeAddr(), 0, 0);

    if (akc->kernarg_preload_spec_length) {
        pendingPreloads.emplace(task->dispatchId(), akc);
    } else {
        delete akc;
    }
    initABI(task);

    ++dynamic_task_id;
    if (!is_blit_kernel) ++non_blit_kernel_id;
}

void
GPUCommandProcessor::sendCompletionSignal(Addr signal_handle, uint16_t vmid,
                                          Event *done_event)
{
    // Originally the completion signal was read functionally and written
    // with a timing DMA. This can cause issues in FullSystem mode and
    // cause translation failures. Therefore, in FullSystem mode everything
    // is done in timing mode.

    if (!FullSystem) {
        /**
        * HACK: The semantics of the HSA signal is to decrement
        * the current signal value. We cheat here and read out
        * he value from main memory using functional access and
        * then just DMA the decremented value.
        */
        uint64_t signal_value = functionalReadHsaSignal(signal_handle);
        updateHsaSignal(signal_handle, signal_value - 1, vmid,
                        [=](const uint64_t &) {
                            if (done_event) {
                                schedule(done_event, curTick());
                            }
                        });
    } else {
        // The semantics of the HSA signal is to decrement the current
        // signal value by one. Do this asynchronously via DMAs and
        // callbacks as we can safely continue with this function
        // while waiting for the next packet from the host.
        updateHsaSignalAsync(signal_handle, -1, vmid, done_event);
    }
}

void
GPUCommandProcessor::updateHsaSignalAsync(Addr signal_handle, int64_t diff,
                                          uint16_t vmid, Event *done_event)
{
    Addr mailbox_addr = getHsaSignalMailboxAddr(signal_handle);
    uint64_t *mailbox_value = new uint64_t;
    auto cb2 = new DmaVirtCallback<uint64_t>([=](const uint64_t &) {
        updateHsaMailboxData(signal_handle, mailbox_value, diff, vmid,
                             done_event);
    });
    dmaReadVirtForVMID(mailbox_addr, sizeof(uint64_t), cb2,
                       (void *)mailbox_value, vmid);
    DPRINTF(GPUCommandProc, "updateHsaSignalAsync reading mailbox addr %lx\n",
            mailbox_addr);
}

void
GPUCommandProcessor::updateHsaMailboxData(Addr signal_handle,
                                          uint64_t *mailbox_value,
                                          int64_t diff, uint16_t vmid,
                                          Event *done_event)
{
    Addr event_addr = getHsaSignalEventAddr(signal_handle);

    DPRINTF(GPUCommandProc, "updateHsaMailboxData read %ld\n", *mailbox_value);
    if (*mailbox_value != 0) {
        // This is an interruptible signal. The mailbox contains the KFD
        // signal-page slot address for this event.
        Addr signal_slot_addr = *mailbox_value;
        auto cb = new DmaVirtCallback<uint64_t>([=](const uint64_t &) {
            updateHsaEventData(signal_handle, signal_slot_addr, mailbox_value,
                               diff, vmid, done_event);
        });
        dmaReadVirtForVMID(event_addr, sizeof(uint64_t), cb,
                           (void *)mailbox_value, vmid);
    } else {
        delete mailbox_value;

        Addr ts_addr = signal_handle + offsetof(amd_signal_t, start_ts);

        amd_event_t *event_ts = new amd_event_t;
        event_ts->start_ts = dispatchStartTime[signal_handle];
        event_ts->end_ts = curTick() / sim_clock::as_int::ns;
        auto cb = new DmaVirtCallback<uint64_t>([=](const uint64_t &) {
            updateHsaEventTs(signal_handle, event_ts, false, 0, diff, vmid,
                             done_event);
        });
        dmaWriteVirtForVMID(ts_addr, sizeof(amd_event_t), cb, (void *)event_ts,
                            vmid);
        DPRINTF(GPUCommandProc, "updateHsaMailboxData reading timestamp addr "
                "%lx\n", ts_addr);

        dispatchStartTime.erase(signal_handle);
        dispatchVMID.erase(signal_handle);
    }
}

void
GPUCommandProcessor::updateHsaEventData(Addr signal_handle,
                                        Addr signal_slot_addr,
                                        uint64_t *event_value, int64_t diff,
                                        uint16_t vmid, Event *done_event)
{
    DPRINTF(GPUCommandProc, "updateHsaEventData read %ld\n", *event_value);

    uint64_t event_id = *event_value;
    uint64_t *event_clear_value = new uint64_t(event_id);
    amd_event_t *event_ts = new amd_event_t;
    event_ts->start_ts = dispatchStartTime[signal_handle];
    event_ts->end_ts = curTick() / sim_clock::as_int::ns;

    auto cb = new DmaVirtCallback<uint64_t>(
        [=](const uint64_t &) { delete event_clear_value; });

    dmaWriteVirtForVMID(signal_slot_addr, sizeof(uint64_t), cb,
                        (void *)event_clear_value, vmid);

    Addr ts_addr = signal_handle + offsetof(amd_signal_t, start_ts);

    auto cb2 = new DmaVirtCallback<uint64_t>([=](const uint64_t &) {
        updateHsaEventTs(signal_handle, event_ts, true, event_id, diff, vmid,
                         done_event);
    });
    dmaWriteVirtForVMID(ts_addr, sizeof(amd_event_t), cb2, (void *)event_ts,
                        vmid);
    DPRINTF(GPUCommandProc, "updateHsaEventData reading timestamp addr %lx\n",
            ts_addr);

    delete event_value;

    dispatchStartTime.erase(signal_handle);
    dispatchVMID.erase(signal_handle);
}

void
GPUCommandProcessor::updateHsaEventTs(Addr signal_handle, amd_event_t *ts,
                                      bool has_event_value, uint64_t event_id,
                                      int64_t diff, uint16_t vmid,
                                      Event *done_event)
{
    delete ts;

    Addr value_addr = getHsaSignalValueAddr(signal_handle);

    uint64_t *signal_value = new uint64_t;
    auto cb = new DmaVirtCallback<uint64_t>([=](const uint64_t &) {
        updateHsaSignalData(value_addr, diff, signal_value, has_event_value,
                            event_id, vmid, done_event);
    });
    dmaReadVirtForVMID(value_addr, sizeof(uint64_t), cb, (void *)signal_value,
                       vmid);
    DPRINTF(GPUCommandProc, "updateHsaSignalAsync reading value addr %lx\n",
            value_addr);
}

void
GPUCommandProcessor::updateHsaSignalData(Addr value_addr, int64_t diff,
                                         uint64_t *prev_value,
                                         bool has_event_value,
                                         uint64_t event_value, uint16_t vmid,
                                         Event *done_event)
{
    // Reuse the value allocated for the read
    DPRINTF(GPUCommandProc, "updateHsaSignalData read %ld, writing %ld\n",
            *prev_value, *prev_value + diff);
    *prev_value += diff;
    auto cb = new DmaVirtCallback<uint64_t>([=](const uint64_t &) {
        if (has_event_value) {
            if (FullSystem) {
                gpuDevice->getIH()->prepareInterruptCookie(
                    static_cast<uint32_t>(event_value), 0,
                    SOC15_IH_CLIENTID_GRBM_CP, CP_EOP, 0, vmid);
                gpuDevice->getIH()->submitInterruptCookie();
            } else {
                signalWakeupEvent(static_cast<uint32_t>(event_value));
            }
        }
        updateHsaSignalDone(prev_value, done_event);
    });
    dmaWriteVirtForVMID(value_addr, sizeof(uint64_t), cb, (void *)prev_value,
                        vmid);
}

void
GPUCommandProcessor::updateHsaSignalDone(uint64_t *signal_value,
                                         Event *done_event)
{
    delete signal_value;

    if (done_event) {
        schedule(done_event, curTick());
    }
}

uint64_t
GPUCommandProcessor::functionalReadHsaSignal(Addr signal_handle)
{
    Addr value_addr = getHsaSignalValueAddr(signal_handle);
    auto tc = system()->threads[0];
    ConstVPtr<Addr> prev_value(value_addr, tc);
    return *prev_value;
}

void
GPUCommandProcessor::updateHsaSignal(Addr signal_handle, uint64_t signal_value,
                                     uint16_t vmid,
                                     HsaSignalCallbackFunction function)
{
    // The signal value is aligned 8 bytes from
    // the actual handle in the runtime
    Addr value_addr = getHsaSignalValueAddr(signal_handle);
    Addr mailbox_addr = getHsaSignalMailboxAddr(signal_handle);
    Addr event_addr = getHsaSignalEventAddr(signal_handle);
    DPRINTF(GPUCommandProc, "Triggering completion signal: %x!\n", value_addr);

    auto tc = system()->threads[0];
    ConstVPtr<uint64_t> mailbox_ptr(mailbox_addr, tc);
    bool mailbox_present = false;
    Addr signal_slot_addr = 0;
    bool signal_wakeup = false;
    uint32_t signal_event = 0;

    // Notifying an event with its mailbox pointer is
    // not supported in the current implementation. Just use
    // mailbox pointer to distinguish between interruptible
    // and default signal. Interruptible signal will have
    // a valid mailbox pointer.
    if (*mailbox_ptr != 0) {
        mailbox_present = true;
        signal_slot_addr = *mailbox_ptr;
        // This is an interruptible signal. Now, read the
        // event ID and directly communicate with the driver
        // about that event notification.
        ConstVPtr<uint32_t> event_val(event_addr, tc);
        signal_event = *event_val;

        DPRINTF(GPUCommandProc,
                "Calling signal wakeup event on "
                "signal event value %d\n",
                signal_event);

        if (!FullSystem) {
            signal_wakeup = true;
        }
    }

    auto cb = new DmaVirtCallback<uint64_t>(
        [=](const uint64_t &dma_buffer) {
            if (FullSystem && mailbox_present) {
                auto mailbox_cb = new DmaVirtCallback<uint64_t>(
                    [=](const uint64_t &) { function(dma_buffer); },
                    static_cast<uint64_t>(signal_event));
                dmaWriteVirtForVMID(signal_slot_addr, sizeof(uint64_t),
                                    mailbox_cb, &mailbox_cb->dmaBuffer, vmid);
                return;
            }
            if (signal_wakeup) {
                signalWakeupEvent(signal_event);
            }
            function(dma_buffer);
        },
        signal_value);

    dmaWriteVirtForVMID(value_addr, sizeof(uint64_t), cb, &cb->dmaBuffer,
                        vmid);
}

void
GPUCommandProcessor::attachDriver(GPUComputeDriver *gpu_driver)
{
    fatal_if(_driver, "Should not overwrite driver.");
    // TODO: GPU Driver inheritance hierarchy doesn't really make sense.
    // Should get rid of the base class.
    _driver = gpu_driver;
    assert(_driver);
}

GPUComputeDriver*
GPUCommandProcessor::driver()
{
    return _driver;
}

/**
 * submitVendorPkt() is for accepting vendor-specific packets from
 * the HSAPP. Vendor-specific packets may be used by the runtime to
 * send commands to the HSA device that are specific to a particular
 * vendor. The vendor-specific packets should be defined by the vendor
 * in the runtime.
 */

/**
 * TODO: For now we simply tell the HSAPP to finish the packet and write a
 * completion signal, if any. However, in the future proper handing may be
 * required for vendor specific packets.
 *
 * In the version of ROCm that is currently supported the runtime will send
 * packets that direct the CP to invalidate the GPU caches. We do this
 * automatically on each kernel launch in the CU, so that situation is safe
 * for now.
 */
void
GPUCommandProcessor::submitVendorPkt(void *raw_pkt, uint32_t queue_id,
    Addr host_pkt_addr)
{
    auto vendor_pkt = (_hsa_generic_vendor_pkt *)raw_pkt;

    if (vendor_pkt->completion_signal) {
        auto *q_desc = hsaPP->getQueueDesc(queue_id);
        auto done = new EventFunctionWrapper(
            [=] { hsaPP->finishPkt(raw_pkt, queue_id); }, name(), true);
        sendCompletionSignal(vendor_pkt->completion_signal, q_desc->vmid,
                             done);
    } else {
        hsaPP->finishPkt(raw_pkt, queue_id);
    }

    warn("Ignoring vendor packet\n");
}

/**
 * submitAgentDispatchPkt() is for accepting agent dispatch packets.
 * These packets will control the dispatch of Wg on the device, and inform
 * the host when a specified number of Wg have been executed on the device.
 *
 * For now it simply finishes the pkt.
 */
void
GPUCommandProcessor::submitAgentDispatchPkt(void *raw_pkt, uint32_t queue_id,
                                            Addr host_pkt_addr,
                                            Event *done_event)
{
    fatal_if(!done_event, "Agent dispatch packet requires a completion event");

    //Parse the Packet, see what it wants us to do
    _hsa_agent_dispatch_packet_t * agent_pkt =
        (_hsa_agent_dispatch_packet_t *)raw_pkt;

    if (agent_pkt->type == AgentCmd::Nop) {
        DPRINTF(GPUCommandProc, "Agent Dispatch Packet NOP\n");
        schedule(done_event, curTick());
    } else if (agent_pkt->type == AgentCmd::Steal) {
        //This is where we steal the HSA Task's completion signal
        int kid = agent_pkt->arg[0];
        DPRINTF(GPUCommandProc,
            "Agent Dispatch Packet Stealing signal handle for kernel %d\n",
            kid);

        HSAQueueEntry *task = dispatcher.hsaTask(kid);
        uint64_t signal_addr = task->completionSignal();// + sizeof(uint64_t);

        uint64_t return_address = agent_pkt->return_address;
        DPRINTF(GPUCommandProc, "Return Addr: %p\n",return_address);
        //*return_address = signal_addr;
        auto cb = new DmaVirtCallback<Addr>(
            [=](const Addr &) { schedule(done_event, curTick()); },
            (Addr)signal_addr);
        dmaWriteVirtForVMID(return_address, sizeof(Addr), cb, &cb->dmaBuffer,
                            task->vmid());

        DPRINTF(GPUCommandProc,
                "Agent Dispatch Packet Stealing signal handle from kid %d :"
                "(%x:%x) writing into %x\n",
                kid, signal_addr, signal_addr, return_address);

    } else
    {
        panic("The agent dispatch packet provided an unknown argument in" \
        "arg[0],currently only 0(nop) or 1(return kernel signal) is accepted");
    }
}

/**
 * Once the CP has finished extracting all relevant information about
 * a task and has initialized the ABI state, we send a description of
 * the task to the dispatcher. The dispatcher will create and dispatch
 * WGs to the CUs.
 */
void
GPUCommandProcessor::dispatchPkt(HSAQueueEntry *task)
{
    dispatcher.dispatch(task);
}

void
GPUCommandProcessor::signalWakeupEvent(uint32_t event_id)
{
    _driver->signalWakeupEvent(event_id);
}

bool
GPUCommandProcessor::preloadReady(HSAQueueEntry *task)
{
    auto it = pendingPreloads.find(task->dispatchId());
    if (it == pendingPreloads.end())
        return true;

    if (it->second) {
        auto *akc = it->second;
        it->second = nullptr;
        readPreload(akc, task);
    }
    return false;
}

void
GPUCommandProcessor::readPreload(AMDKernelCode *akc, HSAQueueEntry *task)
{
    _hsa_dispatch_packet_t *disp_pkt =
        (_hsa_dispatch_packet_t*)task->dispPktPtr();

    // Data preloaded is copied from the kernarg segment. Preloading starts at
    // the dword offset specified by kernarg_preload_spec_offset.
    Addr preload_addr = (Addr)disp_pkt->kernarg_address
        + akc->kernarg_preload_spec_offset * 4;
    const unsigned preload_size =
        sizeof(uint32_t) * akc->kernarg_preload_spec_length;
    panic_if(preload_size > KernargPreloadPktSize,
             "Kernarg preload size %u exceeds local buffer size %u",
             preload_size, KernargPreloadPktSize);

    DPRINTF(GPUCommandProc, "Kernarg preload starts at addr: %#x\n",
            preload_addr);

    /**
     * In full system mode, the page table entry may point to a system
     * page or a device page. System pages use the proxy as normal, but
     * a device page needs to be read from device memory. Check what type
     * it is here.
     */
    bool is_system_page = true;
    Addr phys_addr = preload_addr;

    uint16_t vmid = task->vmid();
    unsigned tmp_bytes;
    walker->startFunctional(gpuDevice->getVM().getPageTableBase(vmid),
                            phys_addr, tmp_bytes, BaseMMU::Mode::Read,
                            is_system_page);

    DPRINTF(GPUCommandProc, "Kernarg preload data is in %s memory\n",
            is_system_page ? "host" : "device");

    /**
     * System objects use DMA device. Device objects need to use device
     * memory.
     */
    if (is_system_page) {
        // Unclear if this is even possible as the point of kernarg preload
        // is to avoid loads from host memory by explicitly placing them in
        // device memory. It is not difficult to implement so issue a warning
        // for now to indicate a possible place to debug if something goes
        // wrong and this warning is seen.
        warn("Preload kernarg from host untested!\n");

        auto cb = new DmaVirtCallback<uint32_t>(
            [ = ] (const uint32_t&) {
                initPreload(akc, task);
            });

        dmaReadVirtForVMID(preload_addr, preload_size, cb, task->preloadArgs(),
                           vmid);
    } else {
        // Read from GPU memory manager one cache line at a time to prevent
        // rare cases where the preload data spans two memory pages.
        constexpr unsigned alignment_granularity = 64;
        ChunkGenerator gen(preload_addr, preload_size, alignment_granularity);

        auto *dispatchReadCtx = new KernelDispatchReadContext;
        dispatchReadCtx->dispatchData.akc = akc;
        dispatchReadCtx->dispatchData.task = task;
        dispatchReadCtx->dispatchType =
            ComputeUnit::SQCPort::SenderState::DISPATCH_PRELOAD_ARG;

        for (; !gen.done(); gen.next()) {
            Addr chunk_addr = gen.addr();
            unsigned dummy;
            walker->startFunctional(
                gpuDevice->getVM().getPageTableBase(vmid), chunk_addr,
                dummy, BaseMMU::Mode::Read, is_system_page);

            Request::Flags flags = Request::PHYSICAL;
            RequestPtr request = std::make_shared<Request>(
                chunk_addr, gen.size(), flags, walker->getDevRequestor());

            PacketPtr readPkt = new Packet(request, MemCmd::ReadReq);
            readPkt->dataStatic((uint8_t *)task->preloadArgs()
                                 + gen.complete());

            ++dispatchReadCtx->pendingReads;
            kernelDispatchReads[readPkt] = dispatchReadCtx;
            performTimingRead(readPkt,
                ComputeUnit::SQCPort::SenderState::DISPATCH_PRELOAD_ARG);
        }
    }
}

void
GPUCommandProcessor::initPreload(AMDKernelCode *akc, HSAQueueEntry *task)
{
    // Fill in SGPRs
    int num_sgprs = akc->kernarg_preload_spec_length;

    task->preloadLength(num_sgprs);
    for (int i = 0; i < num_sgprs; ++i) {
        DPRINTF(GPUCommandProc, "Task preload user SGPR[%d] = %x\n",
                i, task->preloadArgs()[i]);
    }

    pendingPreloads.erase(task->dispatchId());
    delete akc;
    dispatcher.scheduleDispatch();
}

/**
 * The CP is responsible for traversing all HSA-ABI-related data
 * structures from memory and initializing the ABI state.
 * Information provided by the MQD, AQL packet, and code object
 * metadata will be used to initialze register file state.
 */
void
GPUCommandProcessor::initABI(HSAQueueEntry *task)
{
    auto cb = new DmaVirtCallback<uint32_t>(
        [ = ] (const uint32_t &readDispIdOffset)
            { ReadDispIdOffsetDmaEvent(task, readDispIdOffset); }, 0);

    Addr hostReadIdxPtr
        = hsaPP->getQueueDesc(task->queueId())->hostReadIndexPtr;

    uint16_t vmid = hsaPP->getQueueDesc(task->queueId())->vmid;
    dmaReadVirtForVMID(hostReadIdxPtr + sizeof(hostReadIdxPtr),
                       sizeof(uint32_t), cb, &cb->dmaBuffer, vmid);
}

void
GPUCommandProcessor::sanityCheckAKC(AMDKernelCode *akc)
{
    DPRINTF(GPUInitAbi, "group_segment_fixed_size: %d\n",
            akc->group_segment_fixed_size);
    DPRINTF(GPUInitAbi, "private_segment_fixed_size: %d\n",
            akc->private_segment_fixed_size);
    DPRINTF(GPUInitAbi, "kernarg_size: %d\n", akc->kernarg_size);
    DPRINTF(GPUInitAbi, "kernel_code_entry_byte_offset: %d\n",
            akc->kernel_code_entry_byte_offset);
    DPRINTF(GPUInitAbi, "accum_offset: %d\n", akc->accum_offset);
    DPRINTF(GPUInitAbi, "tg_split: %d\n", akc->tg_split);
    DPRINTF(GPUInitAbi, "granulated_workitem_vgpr_count: %d\n",
            akc->granulated_workitem_vgpr_count);
    DPRINTF(GPUInitAbi, "granulated_wavefront_sgpr_count: %d\n",
            akc->granulated_wavefront_sgpr_count);
    DPRINTF(GPUInitAbi, "priority: %d\n", akc->priority);
    DPRINTF(GPUInitAbi, "float_mode_round_32: %d\n", akc->float_mode_round_32);
    DPRINTF(GPUInitAbi, "float_mode_round_16_64: %d\n",
            akc->float_mode_round_16_64);
    DPRINTF(GPUInitAbi, "float_mode_denorm_32: %d\n",
            akc->float_mode_denorm_32);
    DPRINTF(GPUInitAbi, "float_mode_denorm_16_64: %d\n",
            akc->float_mode_denorm_16_64);
    DPRINTF(GPUInitAbi, "priv: %d\n", akc->priv);
    DPRINTF(GPUInitAbi, "enable_dx10_clamp: %d\n", akc->enable_dx10_clamp);
    DPRINTF(GPUInitAbi, "debug_mode: %d\n", akc->debug_mode);
    DPRINTF(GPUInitAbi, "enable_ieee_mode: %d\n", akc->enable_ieee_mode);
    DPRINTF(GPUInitAbi, "bulky: %d\n", akc->bulky);
    DPRINTF(GPUInitAbi, "cdbg_user: %d\n", akc->cdbg_user);
    DPRINTF(GPUInitAbi, "fp16_ovfl: %d\n", akc->fp16_ovfl);
    DPRINTF(GPUInitAbi, "wgp_mode: %d\n", akc->wgp_mode);
    DPRINTF(GPUInitAbi, "mem_ordered: %d\n", akc->mem_ordered);
    DPRINTF(GPUInitAbi, "fwd_progress: %d\n", akc->fwd_progress);
    DPRINTF(GPUInitAbi, "enable_private_segment: %d\n",
            akc->enable_private_segment);
    DPRINTF(GPUInitAbi, "user_sgpr_count: %d\n", akc->user_sgpr_count);
    DPRINTF(GPUInitAbi, "enable_trap_handler: %d\n", akc->enable_trap_handler);
    DPRINTF(GPUInitAbi, "enable_sgpr_workgroup_id_x: %d\n",
            akc->enable_sgpr_workgroup_id_x);
    DPRINTF(GPUInitAbi, "enable_sgpr_workgroup_id_y: %d\n",
            akc->enable_sgpr_workgroup_id_y);
    DPRINTF(GPUInitAbi, "enable_sgpr_workgroup_id_z: %d\n",
            akc->enable_sgpr_workgroup_id_z);
    DPRINTF(GPUInitAbi, "enable_sgpr_workgroup_info: %d\n",
            akc->enable_sgpr_workgroup_info);
    DPRINTF(GPUInitAbi, "enable_vgpr_workitem_id: %d\n",
            akc->enable_vgpr_workitem_id);
    DPRINTF(GPUInitAbi, "enable_exception_address_watch: %d\n",
            akc->enable_exception_address_watch);
    DPRINTF(GPUInitAbi, "enable_exception_memory: %d\n",
            akc->enable_exception_memory);
    DPRINTF(GPUInitAbi, "granulated_lds_size: %d\n", akc->granulated_lds_size);
    DPRINTF(GPUInitAbi, "enable_exception_ieee_754_fp_invalid_operation: %d\n",
            akc->enable_exception_ieee_754_fp_invalid_operation);
    DPRINTF(GPUInitAbi, "enable_exception_fp_denormal_source: %d\n",
            akc->enable_exception_fp_denormal_source);
    DPRINTF(GPUInitAbi, "enable_exception_ieee_754_fp_division_by_zero: %d\n",
            akc->enable_exception_ieee_754_fp_division_by_zero);
    DPRINTF(GPUInitAbi, "enable_exception_ieee_754_fp_overflow: %d\n",
            akc->enable_exception_ieee_754_fp_overflow);
    DPRINTF(GPUInitAbi, "enable_exception_ieee_754_fp_underflow: %d\n",
            akc->enable_exception_ieee_754_fp_underflow);
    DPRINTF(GPUInitAbi, "enable_exception_ieee_754_fp_inexact: %d\n",
            akc->enable_exception_ieee_754_fp_inexact);
    DPRINTF(GPUInitAbi, "enable_exception_int_divide_by_zero: %d\n",
            akc->enable_exception_int_divide_by_zero);
    DPRINTF(GPUInitAbi, "enable_sgpr_private_segment_buffer: %d\n",
            akc->enable_sgpr_private_segment_buffer);
    DPRINTF(GPUInitAbi, "enable_sgpr_dispatch_ptr: %d\n",
            akc->enable_sgpr_dispatch_ptr);
    DPRINTF(GPUInitAbi, "enable_sgpr_queue_ptr: %d\n",
            akc->enable_sgpr_queue_ptr);
    DPRINTF(GPUInitAbi, "enable_sgpr_kernarg_segment_ptr: %d\n",
            akc->enable_sgpr_kernarg_segment_ptr);
    DPRINTF(GPUInitAbi, "enable_sgpr_dispatch_id: %d\n",
            akc->enable_sgpr_dispatch_id);
    DPRINTF(GPUInitAbi, "enable_sgpr_flat_scratch_init: %d\n",
            akc->enable_sgpr_flat_scratch_init);
    DPRINTF(GPUInitAbi, "enable_sgpr_private_segment_size: %d\n",
            akc->enable_sgpr_private_segment_size);
    DPRINTF(GPUInitAbi, "enable_wavefront_size32: %d\n",
            akc->enable_wavefront_size32);
    DPRINTF(GPUInitAbi, "use_dynamic_stack: %d\n", akc->use_dynamic_stack);
    DPRINTF(GPUInitAbi, "kernarg_preload_spec_length: %d\n",
            akc->kernarg_preload_spec_length);
    DPRINTF(GPUInitAbi, "kernarg_preload_spec_offset: %d\n",
            akc->kernarg_preload_spec_offset);


    // Check for features not implemented in gem5
    fatal_if(akc->wgp_mode, "WGP mode not supported\n");
    fatal_if(akc->mem_ordered, "Memory ordering control not supported\n");
    fatal_if(akc->fwd_progress, "Fwd_progress mode not supported\n");


    // Warn on features that gem5 will ignore
    warn_if(akc->fp16_ovfl, "FP16 clamp control bit ignored\n");
    warn_if(akc->bulky, "Bulky code object bit ignored\n");
    // TODO: All the IEEE bits

    warn_if(akc->tg_split, "TG split not implemented\n");
}

System*
GPUCommandProcessor::system()
{
    return sys;
}

AddrRangeList
GPUCommandProcessor::getAddrRanges() const
{
    AddrRangeList ranges;
    return ranges;
}

void
GPUCommandProcessor::setGPUDevice(AMDGPUDevice *gpu_device)
{
    gpuDevice = gpu_device;
    walker->setDevRequestor(gpuDevice->vramRequestorId());
}

void
GPUCommandProcessor::setShader(Shader *shader)
{
    _shader = shader;
}

Shader*
GPUCommandProcessor::shader()
{
    return _shader;
}

GfxVersion
GPUCommandProcessor::getGfxVersion() const
{
    return FullSystem ? gpuDevice->getGfxVersion() : _driver->getGfxVersion();
}

} // namespace gem5
