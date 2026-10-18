#ifndef _LIBKRUN_TRAP_H
#define _LIBKRUN_TRAP_H

#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Trap blobs (MacNeutron, Direct3D forwarding): a dxmt-context blob whose id has bit 63 set stays unmapped in the VM.
 * An 8-byte guest store to it calls fn on the storing vCPU's thread, before the store returns, with the blob's context
 * and the value stored. Set it before the VM starts; NULL removes it. Returns 0.
 */
int32_t krun_set_trap_handler(void (*fn)(uint32_t ctx_id, uint64_t value, void *opaque), void *opaque);

#ifdef __cplusplus
}
#endif

#endif
