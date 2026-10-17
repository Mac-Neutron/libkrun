#ifndef _LIBKRUN_TPM_H
#define _LIBKRUN_TPM_H

#include <inttypes.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The TPM engine behind a TpmDevice (krun_tpm_device_new): the guest's TPM 2.0 commands, one at a time, on a libkrun
 * thread that may change from one command to the next.
 *
 * execute: the command from the guest at `locality` (`len` bytes) into its response, written to `resp` (at most
 * `cap` bytes, 4096), whose length it returns; anything shorter than a response header (10 bytes) answers the guest
 * TPM_RC_FAILURE.
 */
struct krun_tpm_backend {
    void *userdata;
    int32_t (*execute)(void *userdata, uint8_t locality, const uint8_t *cmd, size_t len, uint8_t *resp, size_t cap);
};

#ifdef __cplusplus
}
#endif

#endif
