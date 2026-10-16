/*
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include "dev/amdgpu/amdgpu_vm.hh"

#include <cstring>

#include "arch/amdgpu/vega/pagetable_walker.hh"
#include "arch/amdgpu/vega/tlb.hh"
#include "arch/generic/mmu.hh"
#include "base/trace.hh"
#include "debug/AMDGPUDevice.hh"
#include "dev/amdgpu/amdgpu_defines.hh"
#include "dev/amdgpu/amdgpu_device.hh"
#include "mem/packet_access.hh"
#include "sim/faults.hh"

namespace gem5
{

namespace
{

constexpr Addr kInvalidateEngineCount = 18;

bool
inRegisterRange(Addr offset, Addr last_reg)
{
    return offset >= (last_reg - (kInvalidateEngineCount - 1)) &&
           offset <= last_reg;
}

bool
isInvalidateSemReg(Addr offset)
{
    return inRegisterRange(offset, mmMMHUB_VM_INVALIDATE_ENG17_SEM);
}

bool
isInvalidateAckReg(Addr offset)
{
    return inRegisterRange(offset, mmMMHUB_VM_INVALIDATE_ENG17_ACK) ||
           inRegisterRange(offset, mmVM_INVALIDATE_ENG17_ACK) ||
           inRegisterRange(offset, MI300X_VM_INVALIDATE_ENG17_ACK);
}

} // namespace

AMDGPUVM::AMDGPUVM()
{
    // Zero out contexts
    memset(&vmContext0, 0, sizeof(AMDGPUSysVMContext));

    vmContexts.resize(AMDGPU_VM_COUNT);
    for (int i = 0; i < AMDGPU_VM_COUNT; ++i) {
        memset(&vmContexts[0], 0, sizeof(AMDGPUVMContext));
    }

    for (int i = 0; i < NUM_MMIO_RANGES; ++i) {
        mmioRanges[i] = AddrRange();
    }
}

void
AMDGPUVM::setMMIOAperture(mmio_range_t mmio_aperture, AddrRange range)
{
    mmioRanges[mmio_aperture] = range;
}

AddrRange
AMDGPUVM::getMMIORange(mmio_range_t mmio_aperture)
{
    return mmioRanges[mmio_aperture];
}

const AddrRange&
AMDGPUVM::getMMIOAperture(Addr offset)
{
    for (int i = 0; i < NUM_MMIO_RANGES; ++i) {
        if (mmioRanges[i].contains(offset)) {
            return mmioRanges[i];
        }
    }

    // Default to NBIO
    return mmioRanges[NBIO_MMIO_RANGE];
}

Addr
AMDGPUVM::gartBase()
{
    return vmContext0.ptBase;
}

Addr
AMDGPUVM::gartSize()
{
    return vmContext0.ptEnd - vmContext0.ptStart;
}

void
AMDGPUVM::readMMIO(PacketPtr pkt, Addr offset)
{
    uint32_t value = pkt->getLE<uint32_t>();

    switch (offset) {
      // MMHUB MMIOs
      case mmMMHUB_VM_INVALIDATE_ENG17_SEM:
        DPRINTF(AMDGPUDevice, "Marking invalidate ENG17 SEM acquired\n");
        pkt->setLE<uint32_t>(1);
        break;
      case mmMMHUB_VM_INVALIDATE_ENG17_ACK:
        // This is only used by driver initialization and only expects an ACK
        // for VMID 0 which is the first bit in the response.
        DPRINTF(AMDGPUDevice, "Telling driver invalidate ENG17 is complete\n");
        pkt->setLE<uint32_t>(1);
        break;
      case mmMMHUB_VM_FB_LOCATION_BASE:
        mmhubBase = ((Addr)bits(value, 23, 0) << 24);
        DPRINTF(AMDGPUDevice, "MMHUB FB base set to %#x\n", mmhubBase);
        break;
      case mmMMHUB_VM_FB_LOCATION_TOP:
        mmhubTop = ((Addr)bits(value, 23, 0) << 24) | 0xFFFFFFULL;
        DPRINTF(AMDGPUDevice, "MMHUB FB top set to %#x\n", mmhubTop);
        break;
      // GRBM MMIOs
      case mmVM_INVALIDATE_ENG17_ACK:
      case MI300X_VM_INVALIDATE_ENG17_ACK:
        DPRINTF(AMDGPUDevice, "Overwritting invalidation ENG17 ACK\n");
        pkt->setLE<uint32_t>(1);
        break;
      default:
          if (isInvalidateSemReg(offset)) {
              DPRINTF(AMDGPUDevice, "Marking invalidate SEM %#x acquired\n",
                      offset);
              pkt->setLE<uint32_t>(1);
              break;
          }

          if (isInvalidateAckReg(offset)) {
              DPRINTF(AMDGPUDevice, "Overwriting invalidate ACK %#x\n",
                      offset);
              pkt->setLE<uint32_t>(1);
              break;
          }

        DPRINTF(AMDGPUDevice, "GPUVM read of unknown MMIO %#x\n", offset);
        break;
    }
}

void
AMDGPUVM::writeMMIOGfx900(PacketPtr pkt, Addr offset)
{
    switch (offset) {
      // VMID0 MMIOs
      case mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32:
        vmContext0.ptBaseL = pkt->getLE<uint32_t>();
        // Clear extra bits not part of address
        vmContext0.ptBaseL = insertBits(vmContext0.ptBaseL, 0, 0, 0);
        break;
      case mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32:
        vmContext0.ptBaseH = pkt->getLE<uint32_t>();
        break;
      case mmVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32:
        vmContext0.ptStartL = pkt->getLE<uint32_t>();
        break;
      case mmVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32:
        vmContext0.ptStartH = pkt->getLE<uint32_t>();
        break;
      case mmVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32:
        vmContext0.ptEndL = pkt->getLE<uint32_t>();
        break;
      case mmVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32:
        vmContext0.ptEndH = pkt->getLE<uint32_t>();
        break;
      case mmMC_VM_AGP_TOP: {
        uint32_t val = pkt->getLE<uint32_t>();
        vmContext0.agpTop = (((Addr)bits(val, 23, 0)) << 24) | 0xffffff;
        } break;
      case mmMC_VM_AGP_BOT: {
        uint32_t val = pkt->getLE<uint32_t>();
        vmContext0.agpBot = ((Addr)bits(val, 23, 0)) << 24;
        } break;
      case mmMC_VM_AGP_BASE: {
        uint32_t val = pkt->getLE<uint32_t>();
        vmContext0.agpBase = ((Addr)bits(val, 23, 0)) << 24;
        } break;
      case mmMC_VM_FB_LOCATION_TOP: {
        uint32_t val = pkt->getLE<uint32_t>();
        vmContext0.fbTop = (((Addr)bits(val, 23, 0)) << 24) | 0xffffff;
        } break;
      case mmMC_VM_FB_LOCATION_BASE: {
        uint32_t val = pkt->getLE<uint32_t>();
        vmContext0.fbBase = ((Addr)bits(val, 23, 0)) << 24;
        } break;
      case mmMC_VM_FB_OFFSET: {
        uint32_t val = pkt->getLE<uint32_t>();
        vmContext0.fbOffset = ((Addr)bits(val, 23, 0)) << 24;
        } break;
      case mmMC_VM_SYSTEM_APERTURE_LOW_ADDR: {
        uint32_t val = pkt->getLE<uint32_t>();
        vmContext0.sysAddrL = ((Addr)bits(val, 29, 0)) << 18;
        } break;
      case mmMC_VM_SYSTEM_APERTURE_HIGH_ADDR: {
        uint32_t val = pkt->getLE<uint32_t>();
        vmContext0.sysAddrH = ((Addr)bits(val, 29, 0)) << 18;
        } break;
      default:
        break;
    }
}

void
AMDGPUVM::writeMMIOGfx940(PacketPtr pkt, Addr offset)
{
    switch (offset) {
      // VMID0 MMIOs
      case MI300X_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32:
        vmContext0.ptBaseL = pkt->getLE<uint32_t>();
        // Clear extra bits not part of address
        vmContext0.ptBaseL = insertBits(vmContext0.ptBaseL, 0, 0, 0);
        break;
      case MI300X_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32:
        vmContext0.ptBaseH = pkt->getLE<uint32_t>();
        break;
      case MI300X_CONTEXT0_PAGE_TABLE_START_ADDR_LO32:
        vmContext0.ptStartL = pkt->getLE<uint32_t>();
        break;
      case MI300X_CONTEXT0_PAGE_TABLE_START_ADDR_HI32:
        vmContext0.ptStartH = pkt->getLE<uint32_t>();
        break;
      case MI300X_CONTEXT0_PAGE_TABLE_END_ADDR_LO32:
        vmContext0.ptEndL = pkt->getLE<uint32_t>();
        break;
      case MI300X_CONTEXT0_PAGE_TABLE_END_ADDR_HI32:
        vmContext0.ptEndH = pkt->getLE<uint32_t>();
        break;
      case MI300X_VM_AGP_TOP: {
        uint32_t val = pkt->getLE<uint32_t>();
        vmContext0.agpTop = (((Addr)bits(val, 23, 0)) << 24) | 0xffffff;
        } break;
      case MI300X_VM_AGP_BOT: {
        uint32_t val = pkt->getLE<uint32_t>();
        vmContext0.agpBot = ((Addr)bits(val, 23, 0)) << 24;
        } break;
      case MI300X_VM_AGP_BASE: {
        uint32_t val = pkt->getLE<uint32_t>();
        vmContext0.agpBase = ((Addr)bits(val, 23, 0)) << 24;
        } break;
      case MI300X_VM_FB_LOCATION_TOP: {
        uint32_t val = pkt->getLE<uint32_t>();
        vmContext0.fbTop = (((Addr)bits(val, 23, 0)) << 24) | 0xffffff;
        } break;
      case MI300X_VM_FB_LOCATION_BASE: {
        uint32_t val = pkt->getLE<uint32_t>();
        vmContext0.fbBase = ((Addr)bits(val, 23, 0)) << 24;
        } break;
      case MI300X_VM_FB_OFFSET: {
        uint32_t val = pkt->getLE<uint32_t>();
        vmContext0.fbOffset = ((Addr)bits(val, 23, 0)) << 24;
        } break;
      case MI300X_VM_SYSTEM_APERTURE_LOW_ADDR: {
        uint32_t val = pkt->getLE<uint32_t>();
        vmContext0.sysAddrL = ((Addr)bits(val, 29, 0)) << 18;
        } break;
      case MI300X_VM_SYSTEM_APERTURE_HIGH_ADDR: {
        uint32_t val = pkt->getLE<uint32_t>();
        vmContext0.sysAddrH = ((Addr)bits(val, 29, 0)) << 18;
        } break;
      default:
        break;
    }
}

void
AMDGPUVM::writeMMIO(PacketPtr pkt, Addr offset)
{
    // There are multiple functions due to MMIO addresses being aliased to
    // something different from a previous GFX version. So far this has not
    // been the case for supported MMIO reads but requires special handling
    // for newer gfx942 and gfx950 devices.
    if (gpuDevice->getGfxVersion() == GfxVersion::gfx942 ||
        gpuDevice->getGfxVersion() == GfxVersion::gfx950) {
        writeMMIOGfx940(pkt, offset);
    } else {
        writeMMIOGfx900(pkt, offset);
    }
}

void
AMDGPUVM::registerTLB(VegaISA::GpuTLB *tlb)
{
    DPRINTF(AMDGPUDevice, "Registered a TLB with device\n");
    gpu_tlbs.push_back(tlb);
}

void
AMDGPUVM::invalidateTLBs()
{
    DPRINTF(AMDGPUDevice, "Invalidating all TLBs\n");
    for (auto &tlb : gpu_tlbs) {
        tlb->invalidateAll();
        DPRINTF(AMDGPUDevice, " ... TLB invalidated\n");
    }
}

void
AMDGPUVM::setupWalkerCosim()
{
    DPRINTF(AMDGPUDevice,
            "setupWalkerCosim: %d TLBs registered, vramShmemPtr=%p "
            "vramShmemSize=%#lx\n",
            gpu_tlbs.size(), vramShmemPtr, vramShmemSize);
    for (auto &tlb : gpu_tlbs) {
        tlb->getWalker()->setGpuVM(this);
    }
}

void
AMDGPUVM::serialize(CheckpointOut &cp) const
{
    Addr vm0PTBase = vmContext0.ptBase;
    Addr vm0PTStart = vmContext0.ptStart;
    Addr vm0PTEnd = vmContext0.ptEnd;
    uint64_t gartTableSize;
    SERIALIZE_SCALAR(vm0PTBase);
    SERIALIZE_SCALAR(vm0PTStart);
    SERIALIZE_SCALAR(vm0PTEnd);

    SERIALIZE_SCALAR(vmContext0.agpBase);
    SERIALIZE_SCALAR(vmContext0.agpTop);
    SERIALIZE_SCALAR(vmContext0.agpBot);
    SERIALIZE_SCALAR(vmContext0.fbBase);
    SERIALIZE_SCALAR(vmContext0.fbTop);
    SERIALIZE_SCALAR(vmContext0.fbOffset);
    SERIALIZE_SCALAR(vmContext0.sysAddrL);
    SERIALIZE_SCALAR(vmContext0.sysAddrH);

    SERIALIZE_SCALAR(mmhubBase);
    SERIALIZE_SCALAR(mmhubTop);

    Addr ptBase[AMDGPU_VM_COUNT];
    Addr ptStart[AMDGPU_VM_COUNT];
    Addr ptEnd[AMDGPU_VM_COUNT];
    for (int i = 0; i < AMDGPU_VM_COUNT; i++) {
        ptBase[i] = vmContexts[i].ptBase;
        ptStart[i] = vmContexts[i].ptStart;
        ptEnd[i] = vmContexts[i].ptEnd;
    }
    SERIALIZE_ARRAY(ptBase, AMDGPU_VM_COUNT);
    SERIALIZE_ARRAY(ptStart, AMDGPU_VM_COUNT);
    SERIALIZE_ARRAY(ptEnd, AMDGPU_VM_COUNT);

    gartTableSize = gartTable.size();
    uint64_t* gartTableKey = new uint64_t[gartTableSize];
    uint64_t* gartTableValue = new uint64_t[gartTableSize];
    SERIALIZE_SCALAR(gartTableSize);
    int i = 0;
    for (auto it = gartTable.begin(); it != gartTable.end(); ++it) {
        gartTableKey[i] = it->first;
        gartTableValue[i] = it->second;
        i++;
    }
    SERIALIZE_ARRAY(gartTableKey, gartTableSize);
    SERIALIZE_ARRAY(gartTableValue, gartTableSize);
    delete[] gartTableKey;
    delete[] gartTableValue;
}

void
AMDGPUVM::unserialize(CheckpointIn &cp)
{
    // Unserialize requires fields not be packed
    Addr vm0PTBase;
    Addr vm0PTStart;
    Addr vm0PTEnd;
    uint64_t gartTableSize, *gartTableKey, *gartTableValue;
    UNSERIALIZE_SCALAR(vm0PTBase);
    UNSERIALIZE_SCALAR(vm0PTStart);
    UNSERIALIZE_SCALAR(vm0PTEnd);
    // Clear extra bits not part of the address.
    vmContext0.ptBase = insertBits(vm0PTBase, 0, 0, 0);
    vmContext0.ptStart = vm0PTStart;
    vmContext0.ptEnd = vm0PTEnd;

    UNSERIALIZE_SCALAR(vmContext0.agpBase);
    UNSERIALIZE_SCALAR(vmContext0.agpTop);
    UNSERIALIZE_SCALAR(vmContext0.agpBot);
    UNSERIALIZE_SCALAR(vmContext0.fbBase);
    UNSERIALIZE_SCALAR(vmContext0.fbTop);
    UNSERIALIZE_SCALAR(vmContext0.fbOffset);
    UNSERIALIZE_SCALAR(vmContext0.sysAddrL);
    UNSERIALIZE_SCALAR(vmContext0.sysAddrH);

    UNSERIALIZE_SCALAR(mmhubBase);
    UNSERIALIZE_SCALAR(mmhubTop);

    Addr ptBase[AMDGPU_VM_COUNT];
    Addr ptStart[AMDGPU_VM_COUNT];
    Addr ptEnd[AMDGPU_VM_COUNT];
    UNSERIALIZE_ARRAY(ptBase, AMDGPU_VM_COUNT);
    UNSERIALIZE_ARRAY(ptStart, AMDGPU_VM_COUNT);
    UNSERIALIZE_ARRAY(ptEnd, AMDGPU_VM_COUNT);
    for (int i = 0; i < AMDGPU_VM_COUNT; i++) {
        // Clear extra bits not part of the address.
        vmContexts[i].ptBase = insertBits(ptBase[i], 0, 0, 0);
        vmContexts[i].ptStart = ptStart[i];
        vmContexts[i].ptEnd = ptEnd[i];
    }
    UNSERIALIZE_SCALAR(gartTableSize);
    gartTableKey = new uint64_t[gartTableSize];
    gartTableValue = new uint64_t[gartTableSize];
    UNSERIALIZE_ARRAY(gartTableKey, gartTableSize);
    UNSERIALIZE_ARRAY(gartTableValue, gartTableSize);
    for (uint64_t i = 0; i < gartTableSize; i++) {
        gartTable[gartTableKey[i]] = gartTableValue[i];
    }
    delete[] gartTableKey;
    delete[] gartTableValue;
}

void
AMDGPUVM::AGPTranslationGen::translate(Range &range) const
{
    assert(vm->inAGP(range.vaddr));

    Addr next = roundUp(range.vaddr, AMDGPU_AGP_PAGE_SIZE);
    if (next == range.vaddr)
        next += AMDGPU_AGP_PAGE_SIZE;

    range.size = std::min(range.size, next - range.vaddr);
    range.paddr = range.vaddr - vm->getAGPBot() + vm->getAGPBase();

    DPRINTF(AMDGPUDevice, "AMDGPUVM: AGP translation %#lx -> %#lx\n",
            range.vaddr, range.paddr);
}

void
AMDGPUVM::GARTTranslationGen::translate(Range &range) const
{
    Addr next = roundUp(range.vaddr, AMDGPU_GART_PAGE_SIZE);
    if (next == range.vaddr)
        next += AMDGPU_GART_PAGE_SIZE;
    range.size = std::min(range.size, next - range.vaddr);

    Addr gart_addr = bits(range.vaddr, 63, 12);

    // This table is a bit hard to iterate over. If we cross a page, the next
    // PTE is not necessarily the next entry but actually 7 entries away.
    Addr lsb = bits(gart_addr, 2, 0);
    gart_addr += lsb * 7;

    // GART is a single level translation, so the value at the "virtual" addr
    // is the PTE containing the physical address.
    Addr pte = 0;
    auto result = vm->gartTable.find(gart_addr);
    if (result != vm->gartTable.end()) {
        pte = result->second;
    } else if (vm->vramShmemPtr && vm->gartBase() > 0 &&
               vm->vmContext0.ptStart > 0) {
        // Cosim fallback: read PTE directly from shared VRAM.
        //
        // The GART page table base (ptBase/gartBase) may be stored as
        // either a raw MC byte address or as page_number depending on
        // the GFX generation.  Try the raw-address interpretation
        // first (ptBase IS the VRAM byte offset), then fall back to
        // the page-number interpretation (ptBase << 12 = VRAM offset).
        //
        // The PTE index within the table:
        //   gart_addr = original_page * 8  (after getGARTAddr)
        //   gart_start = ptStart * 8
        //   table_index = gart_addr - gart_start  (byte offset in table)
        Addr gart_start_addr = vm->vmContext0.ptStart * 8;
        if (gart_addr >= gart_start_addr) {
            Addr pte_table_offset = gart_addr - gart_start_addr;

            // Try 1: ptBase is a raw VRAM byte offset
            Addr pte_vram_offset = vm->gartBase() + pte_table_offset;
            if (pte_vram_offset + 8 <= vm->vramShmemSize) {
                memcpy(&pte, vm->vramShmemPtr + pte_vram_offset, sizeof(pte));
            }

            // Try 2: ptBase might be fbBase-relative MC address
            if (pte == 0 && vm->vmContext0.fbBase > 0) {
                Addr adj = vm->gartBase() - vm->vmContext0.fbBase;
                if (adj + pte_table_offset + 8 <= vm->vramShmemSize) {
                    memcpy(&pte, vm->vramShmemPtr + adj + pte_table_offset,
                           sizeof(pte));
                    if (pte != 0) {
                        warn_once("GART cosim: ptBase adjusted by fbBase "
                                  "(%#x - %#x = %#x), PTE=%#x",
                                  vm->gartBase(), vm->vmContext0.fbBase, adj,
                                  pte);
                    }
                }
            }
        }

        // One-shot diagnostic: dump GART table header
        static bool dumpedGart = false;
        if (!dumpedGart) {
            dumpedGart = true;
            warn("GART cosim diag: ptBase=%#x ptStart=%#x ptEnd=%#x "
                 "fbBase=%#x fbTop=%#x fbOffset=%#x "
                 "sysAddrL=%#x sysAddrH=%#x vramSize=%#x",
                 vm->gartBase(), vm->vmContext0.ptStart, vm->vmContext0.ptEnd,
                 vm->vmContext0.fbBase, vm->vmContext0.fbTop,
                 vm->vmContext0.fbOffset, vm->vmContext0.sysAddrL,
                 vm->vmContext0.sysAddrH, vm->vramShmemSize);
            // Dump first 8 PTEs at gartBase
            Addr base = vm->gartBase();
            if (base + 64 <= vm->vramShmemSize) {
                uint64_t ptes[8];
                memcpy(ptes, vm->vramShmemPtr + base, 64);
                warn("GART PTEs at ptBase (%#x):", base);
                for (int i = 0; i < 8; i++) {
                    warn("  PTE[%d] = %#018x", i, ptes[i]);
                }
            }
            // Also dump at ptBase - fbBase if fbBase > 0
            if (vm->vmContext0.fbBase > 0 &&
                vm->gartBase() > vm->vmContext0.fbBase) {
                Addr adj = vm->gartBase() - vm->vmContext0.fbBase;
                if (adj + 64 <= vm->vramShmemSize) {
                    uint64_t ptes[8];
                    memcpy(ptes, vm->vramShmemPtr + adj, 64);
                    warn("GART PTEs at ptBase-fbBase (%#x):", adj);
                    for (int i = 0; i < 8; i++) {
                        warn("  PTE[%d] = %#018x", i, ptes[i]);
                    }
                }
            }
        }
    }

    if (pte != 0) {
        Addr lower_bits = bits(range.vaddr, 11, 0);
        range.paddr = (bits(pte, 47, 12) << 12) | lower_bits;
        warn_once("GART cosim: first successful translation "
                  "vaddr=%#x pte=%#x paddr=%#x",
                  range.vaddr, pte, range.paddr);
    } else {
        // Check if the original (pre-getGARTAddr) address was VRAM.
        // getGARTAddr multiplies page number by 8, so reverse:
        Addr page_num = bits(range.vaddr, 63, 12);
        Addr orig_page = page_num >> 3;
        Addr orig_addr = (orig_page << 12) | bits(range.vaddr, 11, 0);
        if (orig_addr < vm->vramShmemSize && vm->vramShmemPtr) {
            // VRAM address — map to a valid system address so the DMA
            // completes without fault.  The actual VRAM content is in
            // shared memory; we use address 0 (always mapped in gem5)
            // as a sink.  The driver polls VRAM via QEMU, so this
            // write is effectively discarded.
            range.paddr = 0;
            warn_once("GART: VRAM address %#x (orig %#x) mapped to "
                      "sink — VRAM write-backs are no-ops in cosim",
                      range.vaddr, orig_addr);
        } else if (vm->vramShmemPtr) {
            // Cosim mode: map unmapped GART pages to a sink instead of
            // faulting.  Faulting causes an infinite DMA retry loop that
            // crashes gem5.  The actual data will be incorrect (reads
            // return 0, writes are discarded), but this keeps the
            // simulation alive while we diagnose the missing PTE.
            range.paddr = 0;
            warn_once("GART cosim: unmapped page vaddr=%#x "
                      "(gart_addr=%#x gartBase=%#x ptStart=%#x "
                      "fbBase=%#x vramSize=%#x) → sink",
                      range.vaddr, gart_addr, vm->gartBase(),
                      vm->vmContext0.ptStart, vm->vmContext0.fbBase,
                      vm->vramShmemSize);
        } else {
            warn("GART translation for %#x not found (gart_addr=%#x "
                 "gartBase=%#x ptStart=%#x vramSize=%#x hasShmem=%d)",
                 range.vaddr, gart_addr, vm->gartBase(),
                 vm->vmContext0.ptStart, vm->vramShmemSize,
                 vm->vramShmemPtr != nullptr);
            range.paddr = range.vaddr;
            range.fault = std::make_shared<GenericPageTableFault>(range.vaddr);
        }
    }

    DPRINTF(AMDGPUDevice, "GART xlat: vaddr=%#x paddr=%#x (pte=%#x)\n",
            range.vaddr, range.paddr, pte);
}

void
AMDGPUVM::MMHUBTranslationGen::translate(Range &range) const
{
    assert(vm->inMMHUB(range.vaddr));

    Addr next = roundUp(range.vaddr, AMDGPU_MMHUB_PAGE_SIZE);
    if (next == range.vaddr)
        next += AMDGPU_MMHUB_PAGE_SIZE;

    range.size = std::min(range.size, next - range.vaddr);
    // Local offset within this GPU's VRAM
    Addr localOffset = range.vaddr - vm->getMMHUBBase();
    // Translate to global address for Ruby memory routing (multi-GPU)
    range.paddr = vm->gpuDevice->localToGlobalVRAM(localOffset);

    DPRINTF(AMDGPUDevice,
            "AMDGPUVM: MMHUB translation %#lx -> %#lx "
            "(GPU %d, local %#lx)\n",
            range.vaddr, range.paddr, vm->gpuDevice->getGpuId(), localOffset);
}

void
AMDGPUVM::UserTranslationGen::translate(Range &range) const
{
    // Get base address of the page table for this vmid
    Addr base = vm->getPageTableBase(vmid);
    Addr start = vm->getPageTableStart(vmid);
    DPRINTF(AMDGPUDevice, "User tl base %#lx start %#lx walker %p\n",
            base, start, walker);

    bool system_bit;
    unsigned logBytes;
    Addr paddr = range.vaddr;
    Fault fault = walker->startFunctional(base, paddr, logBytes,
                                          BaseMMU::Mode::Read, system_bit);
    if (fault != NoFault) {
        fatal("User translation fault");
    }

    // GPU page size is variable. Use logBytes to determine size.
    const Addr page_size = 1 << logBytes;
    Addr next = roundUp(range.vaddr, page_size);
    if (next == range.vaddr) {
        // We don't know the size of the next page, use default.
        next += AMDGPU_USER_PAGE_SIZE;
    }

    // If we are not in system/host memory, change the address to the MMHUB
    // aperture. This is mapped to the same backing memory as device memory.
    if (!system_bit) {
        paddr += vm->getMMHUBBase();
        assert(vm->inMMHUB(paddr));
    }

    range.size = std::min(range.size, next - range.vaddr);
    range.paddr = paddr;
}

} // namespace gem5
